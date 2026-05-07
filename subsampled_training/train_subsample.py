import copy
import os
import random
import time
from functools import partial, wraps
from typing import Callable, List, Optional
import math
import torch.nn.functional as F

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only, rank_zero_warn
from tqdm.auto import tqdm

import src.models.nn.utils as U
import src.utils as utils
import src.utils.train
from src.dataloaders import SequenceDataset  # TODO make registry
from src.tasks import decoders, encoders, tasks
from src.utils import registry
from src.utils.optim.ema import build_ema_optimizer
from src.utils.optim_groups import add_optimizer_hooks

log = src.utils.train.get_logger(__name__)

# Turn on TensorFloat32 (speeds up large model training substantially)
import torch.backends
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Lots of annoying hacks to get WandbLogger to continuously retry on failure
class DummyExperiment:
    """Dummy experiment."""

    def nop(self, *args, **kw):
        pass

    def __getattr__(self, _):
        return self.nop

    def __getitem__(self, idx) -> "DummyExperiment":
        # enables self.logger.experiment[0].add_image(...)
        return self

    def __setitem__(self, *args, **kwargs) -> None:
        pass


def rank_zero_experiment(fn: Callable) -> Callable:
    """Returns the real experiment on rank 0 and otherwise the DummyExperiment."""

    @wraps(fn)
    def experiment(self):
        @rank_zero_only
        def get_experiment():
            return fn(self)

        return get_experiment() or DummyExperiment()

    return experiment


class CustomWandbLogger(WandbLogger):

    def __init__(self, *args, **kwargs):
        """Modified logger that insists on a wandb.init() call and catches wandb's error if thrown."""

        super().__init__(*args, **kwargs)

    @property
    @rank_zero_experiment
    def experiment(self):
        r"""
        Actual wandb object. To use wandb features in your
        :class:`~pytorch_lightning.core.lightning.LightningModule` do the following.
        Example::
        .. code-block:: python
            self.logger.experiment.some_wandb_function()
        """
        if self._experiment is None:
            if self._offline:
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                while True:
                    try:
                        self._experiment = wandb.init(**self._wandb_init)
                        break
                    except Exception as e:
                        print("wandb Exception:\n", e)
                        t = random.randint(30, 60)
                        print(f"Sleeping for {t} seconds")
                        time.sleep(t)

                # define default x-axis
                if getattr(self._experiment, "define_metric", None):
                    self._experiment.define_metric("trainer/global_step")
                    self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        return self._experiment


class SequenceLightningModule(pl.LightningModule):
    def __init__(self, config):
        # Disable profiling executor. This reduces memory and increases speed.
        try:
            torch._C._jit_set_profiling_executor(False)
            torch._C._jit_set_profiling_mode(False)
        except AttributeError:
            pass

        super().__init__()
        # Passing in config expands it one level, so can access by self.hparams.train instead of self.hparams.config.train
        self.save_hyperparameters(config, logger=False)

        # Dataset arguments
        self.dataset = SequenceDataset.registry[self.hparams.dataset._name_](
            **self.hparams.dataset
        )

        # Check hparams
        self._check_config()

        # PL has some bugs, so add hooks and make sure they're only called once
        self._has_setup = False

        self.setup()  ## Added by KS

    def setup(self, stage=None):
        if not self.hparams.train.disable_dataset:
            self.dataset.setup()

        # We need to set up the model in setup() because for some reason when training with DDP, one GPU uses much more memory than the others
        # In order to not overwrite the model multiple times during different stages, we need this hack
        # TODO PL 1.5 seems to have an option to skip hooks to avoid this
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/5410#issuecomment-762257024
        if self._has_setup:
            return
        else:
            self._has_setup = True

        # Convenience feature: if model specifies encoder, combine it with main encoder
        encoder_cfg = utils.to_list(self.hparams.encoder) + utils.to_list(
            self.hparams.model.pop("encoder", None)
        )
        decoder_cfg = utils.to_list(
            self.hparams.model.pop("decoder", None)
        ) + utils.to_list(self.hparams.decoder)

        # Instantiate model
        self.model = utils.instantiate(registry.model, self.hparams.model)
        if (name := self.hparams.train.post_init_hook['_name_']) is not None:
            kwargs = self.hparams.train.post_init_hook.copy()
            del kwargs['_name_']
            for module in self.modules():
                if hasattr(module, name):
                    getattr(module, name)(**kwargs)

        # Instantiate the task
        self.task = utils.instantiate(
            tasks.registry, self.hparams.task, dataset=self.dataset, model=self.model
        )

        # Create encoders and decoders
        encoder = encoders.instantiate(
            encoder_cfg, dataset=self.dataset, model=self.model
        )
        decoder = decoders.instantiate(
            decoder_cfg, model=self.model, dataset=self.dataset
        )

        # Extract the modules so they show up in the top level parameter count
        self.encoder = U.PassthroughSequential(self.task.encoder, encoder)
        self.decoder = U.PassthroughSequential(decoder, self.task.decoder)
        self.loss = self.task.loss
        self.loss_val = self.task.loss
        if hasattr(self.task, 'loss_val'):
            self.loss_val = self.task.loss_val
        self.metrics = self.task.metrics

        # Handle state logic
        self._initialize_state()

    def load_state_dict(self, state_dict, strict=True):
        if self.hparams.train.pretrained_model_state_hook['_name_'] is not None:
            model_state_hook = utils.instantiate(
                registry.model_state_hook,
                self.hparams.train.pretrained_model_state_hook.copy(),
                partial=True,
            )
            # Modify the checkpoint['state_dict'] inside model_state_hook e.g. to inflate 2D convs to 3D convs
            state_dict = model_state_hook(self.model, state_dict)

        print("Custom load_state_dict function is running.")

        # note, it needs to return something from the normal function we overrided
        return super().load_state_dict(state_dict, strict=strict)

    def _check_config(self):
        assert self.hparams.train.state.mode in [None, "none", "null", "reset", "bptt", "tbptt"]
        assert (
            (n := self.hparams.train.state.n_context) is None
            or isinstance(n, int)
            and n >= 0
        )
        assert (
            (n := self.hparams.train.state.n_context_eval) is None
            or isinstance(n, int)
            and n >= 0
        )

    def _initialize_state(self):
        """Called at model setup and start of epoch to completely reset state"""
        self._state = None
        self._memory_chunks = []

    def _reset_state(self, batch, device=None):
        """Called to construct default_state when necessary, e.g. during BPTT"""
        device = device or batch[0].device
        self._state = self.model.default_state(*batch[0].shape[:1], device=device)

    def _detach_state(self, state):
        if isinstance(state, torch.Tensor):
            return state.detach()
        elif isinstance(state, tuple):
            return tuple(self._detach_state(s) for s in state)
        elif isinstance(state, list):
            return [self._detach_state(s) for s in state]
        elif isinstance(state, dict):
            return {k: self._detach_state(v) for k, v in state.items()}
        elif state is None:
            return None
        else:
            raise NotImplementedError

    def _process_state(self, batch, batch_idx, train=True):
        """Handle logic for state context."""
        # Number of context steps
        key = "n_context" if train else "n_context_eval"
        n_context = self.hparams.train.state.get(key)

        # Don't need to do anything if 0 context steps. Make sure there is no state
        if n_context == 0 and self.hparams.train.state.mode not in ['tbptt']:
            self._initialize_state()
            return

        # Reset state if needed
        if self.hparams.train.state.mode == "reset":
            if batch_idx % (n_context + 1) == 0:
                self._reset_state(batch)

        # Pass through memory chunks
        elif self.hparams.train.state.mode == "bptt":
            self._reset_state(batch)
            with torch.no_grad():  # should be unnecessary because individual modules should handle this
                for _batch in self._memory_chunks:
                    self.forward(_batch)
            # Prepare for next step
            self._memory_chunks.append(batch)
            self._memory_chunks = self._memory_chunks[-n_context:]

        elif self.hparams.train.state.mode == 'tbptt':
            _, _, z = batch
            reset = z["reset"]
            if reset:
                self._reset_state(batch)
            else:
                self._state = self._detach_state(self._state)

    def _on_epoch_start(self):
        self._initialize_state()
        
    def _build_stage_boundaries(self, total_epochs: int, n_stages: int):
        """Return list of stage start epochs, and stage lengths, splitting as evenly as possible."""
        assert n_stages >= 1
        base = total_epochs // n_stages
        rem = total_epochs % n_stages
        lengths = [(base + 1 if i < rem else base) for i in range(n_stages)]
        starts = [0]
        for L in lengths[:-1]:
            starts.append(starts[-1] + L)
        return starts, lengths

    def _get_stage_info(self, epoch: int):
        """
        Returns:
        stage_idx: int in [0, n_stages-1]
        stage_start: epoch index where this stage starts
        stage_len: number of epochs in this stage
        """
        n_stages = int(self.hparams.train.get("subsample_n_stages", 2))
        total_epochs = int(getattr(self.trainer, "max_epochs", 0) or self.hparams.trainer.get("max_epochs"))
        starts, lens = self._build_stage_boundaries(total_epochs, n_stages)

        stage_idx = 0
        for i in range(n_stages):
            if epoch >= starts[i]:
                stage_idx = i
        return stage_idx, starts[stage_idx], lens[stage_idx], n_stages

    def _stride_and_scale_dt(self, epoch: int):
        """
        Stage k (0-indexed) uses stride = 2^(n-1-k).
        scale_dt follows your earlier convention: scale_dt = stride / max_stride,
        so it halves each time stride halves.
        """
        stage_idx, stage_start, stage_len, n_stages = self._get_stage_info(epoch)
        stride = 2 ** (n_stages - 1 - stage_idx)
        max_stride = 2 ** (n_stages - 1)
        scale_dt = float(stride) / float(max_stride)
        return stride, scale_dt, stage_idx

    def _mean_subsample(self, x: torch.Tensor, stride: int) -> torch.Tensor:
        """
        Mean-pool subsampling along sequence length dimension.
        Assumes x is (B, L, ...) with L at dim=1.
        Casts integer inputs to float for pooling.
        """
        if stride <= 1:
            return x

        # mean requires float/complex
        if not torch.is_floating_point(x) and not torch.is_complex(x):
            x = x.float()

        B, L = x.shape[0], x.shape[1]
        pad = (stride - (L % stride)) % stride
        if pad:
            orig_shape = x.shape
            x3 = x.reshape(B, L, -1)          # (B, L, Dflat)
            x3 = F.pad(x3, (0, 0, 0, pad))    # pad length dim
            x = x3.reshape(B, L + pad, *orig_shape[2:])

        Lp = x.shape[1]
        newL = Lp // stride
        x3 = x.reshape(B, newL, stride, *x.shape[2:])
        return x3.mean(dim=2)

    def forward(self, batch):
        """Passes a batch through the encoder, backbone, and decoder"""
        x, y, *z = batch
        if len(z) == 0:
            z = {}
        else:
            assert len(z) == 1 and isinstance(z[0], dict), "Dataloader must return dictionary of extra arguments"
            z = z[0]

        # Encoder (embedding/feature map) FIRST
        x, w = self.encoder(x, **z)

        # ---- NEW: stage-based mean subsampling AFTER embedding ----
        stride, scale_dt, stage_idx = self._stride_and_scale_dt(self.current_epoch)

        # x could be (B, L, D) or (B, D, L) depending on encoder; handle both
        if x.dim() >= 3:
            if x.shape[1] == x.shape[-1]:
                # ambiguous; assume (B, L, D)
                x = self._mean_subsample(x, stride=stride)
            else:
                # heuristic: if x is (B, D, L), pool along last dim
                # implement a second helper if needed; simplest is transpose -> pool -> transpose back
                if x.shape[1] < x.shape[-1]:  # common for (B, D, L) where L is large
                    x = x.transpose(-1, -2)           # (B, L, D)
                    x = self._mean_subsample(x, stride=stride)
                    x = x.transpose(-1, -2)           # (B, D, L)
                else:
                    # default: treat as (B, L, D)
                    x = self._mean_subsample(x, stride=stride)
        else:
            # if somehow not a sequence tensor, skip
            pass

        # Backbone (assumes it supports scale_dt=...)
        x, state = self.model(x, scale_dt=scale_dt, **w, state=self._state)
        self._state = state

        # Decoder
        x, w = self.decoder(x, state=state, **z)
        return x, y, w

    def step(self, x_t):
        x_t, *_ = self.encoder(x_t) # Potential edge case for encoders that expect (B, L, H)?
        x_t, state = self.model.step(x_t, state=self._state)
        self._state = state
        # x_t = x_t[:, None, ...] # Dummy length
        # x_t, *_ = self.decoder(x_t, state=state)
        # x_t = x_t[:, 0, ...]
        x_t, *_ = self.decoder.step(x_t, state=state)
        return x_t

    def _shared_step(self, batch, batch_idx, prefix="train"):

        self._process_state(batch, batch_idx, train=(prefix == "train"))

        x, y, w = self.forward(batch)

        # Loss
        if prefix == 'train':
            loss = self.loss(x, y, **w)
        else:
            loss = self.loss_val(x, y, **w)

        # Metrics
        metrics = self.metrics(x, y, **w)
        metrics["loss"] = loss
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}

        # Calculate torchmetrics: these are accumulated and logged at the end of epochs
        self.task.torchmetrics(x, y, prefix)

        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_start(self):
        self._on_epoch_start()
        # Reset training torchmetrics
        self.task._reset_torchmetrics("train")

    def on_train_epoch_end(self):
        # Log training torchmetrics
        super().on_train_epoch_end()
        self.log_dict(
            {f"train/{k}": v for k, v in self.task.get_torchmetrics("train").items()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

    def on_validation_epoch_start(self):
        self._on_epoch_start()
        # Reset all validation torchmetrics
        for name in self.val_loader_names:
            self.task._reset_torchmetrics(name)

    def on_validation_epoch_end(self):
        # Log all validation torchmetrics
        super().on_validation_epoch_end()
        for name in self.val_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def on_test_epoch_start(self):
        self._on_epoch_start()
        # Reset all test torchmetrics
        for name in self.test_loader_names:
            self.task._reset_torchmetrics(name)

    def on_test_epoch_end(self):
        # Log all test torchmetrics
        super().on_test_epoch_end()
        for name in self.test_loader_names:
            self.log_dict(
                {f"{name}/{k}": v for k, v in self.task.get_torchmetrics(name).items()},
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                add_dataloader_idx=False,
                sync_dist=True,
            )

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, prefix="train")

        # Log the loss explicitly so it shows up in WandB
        # Note that this currently runs into a bug in the progress bar with ddp (as of 1.4.6)
        # https://github.com/PyTorchLightning/pytorch-lightning/pull/9142
        # We additionally log the epochs under 'trainer' to get a consistent prefix with 'global_step'
        loss_epoch = {"trainer/loss": loss, "trainer/epoch": self.current_epoch}
        self.log_dict(
            loss_epoch,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        # Log any extra info that the models want to expose (e.g. output norms)
        metrics = {}
        for module in list(self.modules())[1:]:
            if hasattr(module, "metrics"):
                metrics.update(module.metrics)

        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            add_dataloader_idx=False,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        ema = (
            self.val_loader_names[dataloader_idx].endswith("/ema")
            and self.optimizers().optimizer.stepped
        )  # There's a bit of an annoying edge case with the first (0-th) epoch; it has to be excluded due to the initial sanity check
        if ema:
            self.optimizers().swap_ema()
        loss = self._shared_step(
            batch, batch_idx, prefix=self.val_loader_names[dataloader_idx]
        )
        if ema:
            self.optimizers().swap_ema()

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self._shared_step(
            batch, batch_idx, prefix=self.test_loader_names[dataloader_idx]
        )

    def configure_optimizers(self):

        # Set zero weight decay for some params
        if 'optimizer_param_grouping' in self.hparams.train:
            add_optimizer_hooks(self.model, **self.hparams.train.optimizer_param_grouping)

        # Normal parameters
        all_params = list(self.parameters())
        params = [p for p in all_params if not hasattr(p, "_optim")]


        # Construct optimizer, add EMA if necessary
        if self.hparams.train.ema > 0.0:
            optimizer = utils.instantiate(
                registry.optimizer,
                self.hparams.optimizer,
                params,
                wrap=build_ema_optimizer,
                polyak=self.hparams.train.ema,
            )
        else:
            optimizer = utils.instantiate(registry.optimizer, self.hparams.optimizer, params)

        del self.hparams.optimizer._name_

        # Add parameters with special hyperparameters
        hps = [getattr(p, "_optim") for p in all_params if hasattr(p, "_optim")]
        hps = [
            # dict(s) for s in set(frozenset(hp.items()) for hp in hps)
            dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))
            # dict(s) for s in dict.fromkeys(frozenset(hp.items()) for hp in hps)
        ]  # Unique dicts
        print("Hyperparameter groups", hps)
        for hp in hps:
            params = [p for p in all_params if getattr(p, "_optim", None) == hp]
            optimizer.add_param_group(
                {"params": params, **self.hparams.optimizer, **hp}
            )

        ### Layer Decay ###

        if self.hparams.train.layer_decay['_name_'] is not None:
            get_num_layer = utils.instantiate(
                registry.layer_decay,
                self.hparams.train.layer_decay['_name_'],
                partial=True,
            )

            # Go through all parameters and get num layer
            layer_wise_groups = {}
            num_max_layers = 0
            for name, p in self.named_parameters():
                # Get layer id for each parameter in the model
                layer_id = get_num_layer(name)

                # Add to layer wise group
                if layer_id not in layer_wise_groups:
                    layer_wise_groups[layer_id] = {
                        'params': [],
                        'lr': None,
                        'weight_decay': self.hparams.optimizer.weight_decay
                    }
                layer_wise_groups[layer_id]['params'].append(p)

                if layer_id > num_max_layers: num_max_layers = layer_id

            # Update lr for each layer
            for layer_id, group in layer_wise_groups.items():
                group['lr'] = self.hparams.optimizer.lr * (self.hparams.train.layer_decay.decay ** (num_max_layers - layer_id))

            # Reset the torch optimizer's param groups
            optimizer.param_groups = []
            for layer_id, group in layer_wise_groups.items():
                optimizer.add_param_group(group)

        # Print optimizer info for debugging
        keys = set([k for hp in hps for k in hp.keys()])  # Special hparams
        utils.train.log_optimizer(log, optimizer, keys)

        # ---- NEW: step-based warmup + stage-wise cosine restarts ----
        interval = "step"

        # Read warmup steps from config; no more train.warmup
        warmup_steps = 0
        if "scheduler" in self.hparams and self.hparams.scheduler is not None:
            warmup_steps = int(getattr(self.hparams.scheduler, "num_warmup_steps", 0) or 0)

        def _safe_steps_per_epoch():
            """Get a finite steps/epoch. Raises a helpful error if it’s infinite/unknown."""
            nb = getattr(self.trainer, "num_training_batches", None)
            if isinstance(nb, (int, float)) and math.isfinite(nb) and nb > 0:
                return int(nb)

            # Try dataloader len
            dl = getattr(self.trainer, "train_dataloader", None)
            if dl is not None:
                try:
                    return max(1, len(dl))
                except TypeError:
                    pass

            raise RuntimeError(
                "Cannot infer steps_per_epoch (trainer.num_training_batches is inf or dataloader has no __len__). "
                "Stage-wise schedules need finite epoch length. "
                "Fix by using a finite-length DataLoader or set limit_train_batches to a finite value."
            )

        def lr_lambda(step: int):
            """
            step = optimizer step count (LambdaLR uses last_epoch as step index).
            Returns LR multiplier in [0, 1].
            Assumption: warmup_steps ends before stage 0 ends.
            """
            step = int(step)

            # 1) Warmup (global, only at very beginning)
            if warmup_steps > 0 and step < warmup_steps:
                return max(0.0, min(1.0, step / float(warmup_steps)))

            # 2) Stage-wise cosine restart in *steps*
            steps_per_epoch = _safe_steps_per_epoch()

            # Figure out current epoch (approx) and stage
            epoch = int(step // steps_per_epoch)

            stage_idx, stage_start_epoch, stage_len_epoch, n_stages = self._get_stage_info(epoch)

            stage_start_step = int(stage_start_epoch * steps_per_epoch)
            stage_len_steps = max(1, int(stage_len_epoch * steps_per_epoch))

            # Stage-local step
            t = step - stage_start_step  # steps into stage

            if stage_idx == 0 and warmup_steps > 0:
                # In stage 0, cosine starts AFTER warmup
                denom = max(1, stage_len_steps - warmup_steps)
                t_eff = max(0, step - warmup_steps)  # steps since warmup ended
            else:
                denom = stage_len_steps
                t_eff = t

            # clamp into [0, denom]
            t_eff = max(0.0, min(float(denom), float(t_eff)))

            return 0.5 * (1.0 + math.cos(math.pi * (t_eff / float(denom))))

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        scheduler = {
            "scheduler": lr_scheduler,
            "interval": interval,     # step-based
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_loader = self.dataset.train_dataloader(**self.hparams.loader)
        # Print stats in a try block since some dataloaders might not have a length?
        try:
            log.info(
                f"Loaded 'train' dataloader:".ljust(30) +
                f"{len(train_loader.dataset):7} examples | {len(train_loader):6} steps"
            )
        except:
            pass
        return train_loader

    def _eval_dataloaders_names(self, loaders, prefix):
        """Process loaders into a list of names and loaders"""
        if utils.is_dict(loaders):
            return [
                f"{prefix}/{k}" if k is not None else prefix for k in loaders.keys()
            ], list(loaders.values())
        elif utils.is_list(loaders):
            return [f"{prefix}/{i}" for i in range(len(loaders))], loaders
        else:
            return [prefix], [loaders]

    def _eval_dataloaders(self):
        # Return all val + test loaders
        val_loaders = self.dataset.val_dataloader(**self.hparams.loader)
        test_loaders = self.dataset.test_dataloader(**self.hparams.loader)
        val_loader_names, val_loaders = self._eval_dataloaders_names(val_loaders, "val")
        test_loader_names, test_loaders = self._eval_dataloaders_names(
            test_loaders, "test"
        )

        # Duplicate datasets for ema
        if self.hparams.train.ema > 0.0:
            val_loader_names += [name + "/ema" for name in val_loader_names]
            val_loaders = val_loaders + val_loaders
            test_loader_names += [name + "/ema" for name in test_loader_names]
            test_loaders = test_loaders + test_loaders

        # adding option to only have val loader at eval (eg if test is duplicate)
        if self.hparams.train.get("remove_test_loader_in_eval", None) is not None:
            eval_loader_names = val_loader_names
            eval_loaders = val_loaders
        # default behavior is to add test loaders in eval
        else:
            eval_loader_names = val_loader_names + test_loader_names
            eval_loaders = val_loaders + test_loaders

        return eval_loader_names, eval_loaders

    def val_dataloader(self):
        val_loader_names, val_loaders = self._eval_dataloaders()
        self.val_loader_names = val_loader_names
        try:
            for name, loader in zip(val_loader_names, val_loaders):
                log.info(
                    f"Loaded '{name}' dataloader:".ljust(30) +
                    f"{len(loader.dataset):7} examples | {len(loader):6} steps"
                )
        except:
            pass

        return val_loaders

    def test_dataloader(self):
        test_loader_names, test_loaders = self._eval_dataloaders()
        self.test_loader_names = ["final/" + name for name in test_loader_names]
        return test_loaders


### pytorch-lightning utils and entrypoint ###

def create_trainer(config):
    callbacks: List[pl.Callback] = []
    logger = None

    # WandB Logging
    if config.get("wandb") is not None:
        # Pass in wandb.init(config=) argument to get the nice 'x.y.0.z' hparams logged
        # Can pass in config_exclude_keys='wandb' to remove certain groups
        import wandb

        logger = CustomWandbLogger(
            config=utils.to_dict(config, recursive=True),
            settings=wandb.Settings(start_method="fork"),
            **config.wandb,
        )

    # Lightning callbacks
    if "callbacks" in config:
        for _name_, callback in config.callbacks.items():
            if callback is None: continue
            if config.get("wandb") is None and _name_ in ["learning_rate_monitor"]:
                continue
            log.info(f"Instantiating callback <{registry.callbacks[_name_]}>")
            callback._name_ = _name_
            callbacks.append(utils.instantiate(registry.callbacks, callback))

    # Profiler
    profiler = None
    if config.trainer.get("profiler", None) is not None:
        profiler = hydra.utils.instantiate(config.trainer.profiler)
        config.trainer.pop("profiler")


    # Configure ddp automatically
    if config.trainer.accelerator == 'gpu' and config.trainer.devices > 1:
        print("ddp automatically configured, more than 1 gpu used!")
        config.trainer.strategy = "ddp"

    # Add ProgressiveResizing callback
    if config.callbacks.get("progressive_resizing", None) is not None:
        num_stages = len(config.callbacks.progressive_resizing.stage_params)
        print(f"Progressive Resizing: {num_stages} stages")
        for i, e in enumerate(config.callbacks.progressive_resizing.stage_params):
            # Stage params are resolution and epochs, pretty print
            print(f"\tStage {i}: {e['resolution']} @ {e['epochs']} epochs")

    # Additional ModelCheckpoint callback for preemption
    if config.tolerance.id is not None:
        pass

    trainer = pl.Trainer(
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        **config.trainer,
    )
    return trainer


def train(config):
    if config.train.seed is not None:
        pl.seed_everything(config.train.seed, workers=True)
    trainer = create_trainer(config)
    model = SequenceLightningModule(config)

    # Load pretrained_model if specified
    if config.train.get("pretrained_model_path", None) is not None:
        # PTL style.  Note, method returns a new model object, and need to pass config.
        model = SequenceLightningModule.load_from_checkpoint(
            config.train.pretrained_model_path,
            config=config,
            strict=config.train.pretrained_model_strict_load,
        )
        print("Loaded pretrained model from", config.train.pretrained_model_path)

        # Added by KS for pre-training
        # [22-07-21 AG] refactored, untested
        if config.train.get("ignore_pretrained_layers", False):
            pretrained_dict = pretrained_model.state_dict()
            model_dict = model.state_dict()
            for k, v in model_dict.items():
                for ignore_layer in config.train.ignore_pretrained_layers:
                    if ignore_layer in k:
                        pretrained_dict[k] = v
            model.load_state_dict(pretrained_dict)
        if config.train.get("pretrained_freeze_encoder", False):
            for name, param in model.named_parameters():
                if not("decoder" in name): param.requires_grad = False


    # Run initial validation epoch (useful for debugging, finetuning)
    if config.train.validate_at_start:
        print("Running validation before training")
        trainer.validate(model)

    if config.train.ckpt is not None:
        trainer.fit(model, ckpt_path=config.train.ckpt)
    else:
        trainer.fit(model)
    if config.train.test:
        trainer.test(model)



def preemption_setup(config):
    if config.tolerance.id is None:
        return config

    # Create path ./logdir/id/ to store information for resumption
    resume_dir = os.path.join(get_original_cwd(), config.tolerance.logdir, str(config.tolerance.id))

    if os.path.exists(resume_dir):
        print(f"Resuming from {resume_dir}")

        # Load path to the last checkpoint
        with open(os.path.join(resume_dir, "hydra.txt"), "r") as f:
            hydra_paths = list(f.readlines())

        # Look at the previous runs in reverse order
        checkpoint_path = None
        for hydra_path in reversed(hydra_paths):
            hydra_path = hydra_path.rstrip('\n')

            # Get the paths to the last.ckpt and last_.ckpt files
            last_path = os.path.join(hydra_path, "checkpoints", "last.ckpt")
            # last__path = os.path.join(hydra_path, "checkpoints", "last_.ckpt")
            # last_exists, last__exists = os.path.exists(last_path), os.path.exists(last__path)

            if os.path.exists(last_path):
                print("\tFound checkpoint at", last_path)
                config.train.ckpt = last_path
                # HACK TODO
                config.train.pretrained_model_path = None
                config.train.pretrained_model_state_hook._name_ = None
                # config.train.pretrained_model_reinit_hook._name_ = None
                break

        # If we didn't find a checkpoint
        if checkpoint_path is None:
            print("\tNo suitable checkpoint found, starting from scratch")

        # Set wandb run id to resume
        if os.path.exists(os.path.join(hydra_path, 'wandb')):
            run_info = [e for e in os.listdir(os.path.join(hydra_path, 'wandb')) if e.startswith('run-')][0]
            run_id = run_info.split('-')[-1]
            try:
                config.wandb.id = run_id
            except AttributeError:
                pass

    os.makedirs(resume_dir, exist_ok=True)

    # Store path to Hydra output folder
    with open(os.path.join(resume_dir, 'hydra.txt'), 'a') as f:
        f.write(os.getcwd() + '\n')

    return config


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: OmegaConf):

    # Process config:
    # - register evaluation resolver
    # - filter out keys used only for interpolation
    # - optional hooks, including disabling python warnings or debug friendly configuration
    config = utils.train.process_config(config)

    # Pretty print config using Rich library
    utils.train.print_config(config, resolve=True)

    config = preemption_setup(config)

    train(config)


if __name__ == "__main__":
    main()
