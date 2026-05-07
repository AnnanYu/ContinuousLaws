'''
Adapted from the S4 script https://github.com/state-spaces/s4/
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import os
import argparse

import numpy as np
import time

import math

from s4d import S4D
from tqdm.auto import tqdm
from dataloading import Datasets

import numpy as np

# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# Optimizer
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--dt_min', default=0.0001, type=float, help='min dt')
parser.add_argument('--dt_max', default=0.1, type=float, help='max dt')
parser.add_argument('--weight_decay', default=0.03, type=float, help='Weight decay')
parser.add_argument('--path', default='ckpt', type=str, help='store path')
# Scheduler
parser.add_argument('--epochs', default=80, type=int, help='Training epochs')
# Dataset
parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'cifar10'], type=str, help='Dataset')
parser.add_argument('--grayscale', action='store_false', help='Use grayscale CIFAR10')
# Dataloader
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers to use for dataloader')
parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
# Model
parser.add_argument('--n_layers', default=6, type=int, help='Number of layers')
parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
parser.add_argument('--dropout', default=0.0, type=float, help='Dropout')
parser.add_argument('--prenorm', action='store_false', help='Prenorm')
parser.add_argument('--val_acc_dir', default="", type=str, help='Location to save validation accuracies')
# General
parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
parser.add_argument('--warmup', default=10, type=int)

args = parser.parse_args()

import sys
print(sys.argv[1:])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print(f'==> Preparing {args.dataset} data..')

create_dataset_fn = Datasets["pathx-classification"]
trainloader, valloader, testloader, aux_dataloaders, n_classes, seq_len, in_dim, train_size = \
      create_dataset_fn(seed=231002, bsz=args.batch_size)
d_input = 1
d_output = n_classes

class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=256,
        n_layers=4,
        dropout=0.1,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, dropout=dropout, transposed=True, lr=min(0.00004, args.lr), dt_min = args.dt_min, dt_max = args.dt_max)
            )
            self.norms.append(nn.BatchNorm1d(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x, scale_dt=1):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        x = torch.cat((x,torch.flip(x,dims=[-1])),dim=-1) # Make it bidirectional
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z, scale_dt = scale_dt)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x)

        x = x.transpose(-1, -2)

        # Pooling: average pooling over the sequence length
        x = x.mean(dim=1)

        # Decode the outputs
        x = self.decoder(x)  # (B, d_model) -> (B, d_output)

        return x

# Model
print('==> Building model..')
model = S4Model(
    d_input=d_input,
    d_output=d_output,
    d_model=args.d_model,
    n_layers=args.n_layers,
    dropout=args.dropout,
    prenorm=args.prenorm,
)

model = model.to(device)
if device == 'cuda':
    cudnn.benchmark = True
# -----------------------------------------------------------------------------
# Sub-sampling curriculum (stride 8 -> 4 -> 2 -> 1), 20 epochs per stage.
# When sub-sampling by `stride`, we increase the effective timestep by `stride`,
# so we pass `scale_dt=stride` into S4D (dt = exp(log_dt) * scale_dt).
# -----------------------------------------------------------------------------
STAGE_EPOCHS = 27
STRIDE_SCHEDULE = (4, 2, 1)

def curriculum(epoch: int):
    """Return (stride, scale_dt) for the given epoch."""
    stage = min(epoch // STAGE_EPOCHS, len(STRIDE_SCHEDULE) - 1)
    stride = int(STRIDE_SCHEDULE[stage])
    scale_dt = float(stride)
    return stride, scale_dt

def subsample_inputs(x: torch.Tensor, stride: int) -> torch.Tensor:
    """
    x: (B, L, d_input)
    returns: (B, ceil(L/stride), d_input) by averaging over windows of size `stride`.
            The last window may be shorter and is averaged over its actual length.
    """
    if stride == 1:
        return x

    B, L, D = x.shape
    # Pad to a multiple of stride by repeating the last element (so shape works out)
    pad = (-L) % stride
    if pad:
        x = torch.cat([x, x[:, -1:, :].expand(B, pad, D)], dim=1)  # (B, L+pad, D)

    xw = x.view(B, (L + pad) // stride, stride, D)  # (B, T, stride, D)

    if pad == 0:
        return xw.mean(dim=2)

    # If we padded, correct the last window’s mean to use true count (stride - pad)
    out = xw.mean(dim=2)  # (B, T, D) provisional
    true_len_last = stride - pad
    # recompute last block mean without counting padded repeats too much
    last_block = x[:, (L + pad) - stride : (L + pad), :]  # this includes padded repeats
    last_block_true = last_block[:, :true_len_last, :]    # only original part
    out[:, -1, :] = last_block_true.mean(dim=1)
    return out


def warmup_cosine_annealing_lr_step(step, warmup_steps, total_steps, lr_start=1.0, lr_end=0.0):
    """
    Multiplicative factor for base lr.

    IMPORTANT: This version guarantees lr == lr_end on the *last* step (step == total_steps-1),
    assuming total_steps >= 2 and warmup_steps < total_steps.
    """
    step = int(step)
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(int(warmup_steps), 0)

    # Clamp to [0, total_steps-1]
    step = max(0, min(step, total_steps - 1))

    # Warmup: linear 0 -> lr_start
    if warmup_steps > 0 and step < warmup_steps:
        return lr_start * (step / max(1, warmup_steps))

    # If no room for cosine, just return lr_end
    if total_steps <= warmup_steps + 1:
        return lr_end

    # Cosine over steps [warmup_steps, total_steps-1] inclusive
    denom = (total_steps - 1) - warmup_steps
    t = (step - warmup_steps) / denom  # in [0,1]
    return lr_end + (lr_start - lr_end) * 0.5 * (1.0 + math.cos(math.pi * t))


def setup_optimizer(model, lr, weight_decay, epochs, warmup_epochs, steps_per_epoch, stage_epochs):
    all_parameters = list(model.parameters())

    # General params
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    # Special param groups (S4 params with p._optim)
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        group_params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": group_params, **hp})

    # ----- stage settings -----
    stage_steps = int(stage_epochs * steps_per_epoch)
    stage_steps = max(stage_steps, 1)

    warmup_steps_global = int(warmup_epochs * steps_per_epoch)  # only applied in stage 0

    def lr_lambda(global_step: int):
        # which stage are we in?
        stage_idx = global_step // stage_steps
        s = global_step % stage_steps  # step within stage

        if stage_idx == 0:
            # stage 0: warmup + cosine to 0
            warmup_steps = min(warmup_steps_global, max(stage_steps - 1, 0))
            return warmup_cosine_annealing_lr_step(
                s, warmup_steps, stage_steps, lr_start=1.0, lr_end=0.0
            )
        else:
            # later stages: no warmup, cosine restart from 1 -> 0
            return warmup_cosine_annealing_lr_step(
                s, warmup_steps=0, total_steps=stage_steps, lr_start=1.0, lr_end=0.0
            )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)

    print(
        f"[Scheduler] stage cosine-to-0 with warmup only in stage0: "
        f"stage_steps={stage_steps}, warmup_steps_stage0={warmup_steps_global}"
    )
    return optimizer, scheduler


criterion = nn.CrossEntropyLoss()
steps_per_epoch = len(trainloader)
optimizer, scheduler = setup_optimizer(
    model,
    lr=args.lr,
    weight_decay=args.weight_decay,
    epochs=args.epochs,
    warmup_epochs=args.warmup,
    steps_per_epoch=steps_per_epoch,
    stage_epochs=STAGE_EPOCHS,
)


if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/' + args.path + '.pth')
    model.load_state_dict(checkpoint['model'])
    best_acc = checkpoint['acc']
    val_acc = best_acc
    start_epoch = checkpoint['epoch'] + 1
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

###############################################################################
# Everything after this point is standard PyTorch training!
###############################################################################

# Training
def train(stride: int, scale_dt: float):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(trainloader))
    for batch_idx, (inputs, targets, _) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = subsample_inputs(inputs, stride)
        optimizer.zero_grad()
        outputs = model(inputs, scale_dt=scale_dt)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(
            'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %.3e' %
            (batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total, correct, total, optimizer.param_groups[0]['lr'])
        )
        
    return 100.*correct/total


def eval(epoch, dataloader, stride: int, scale_dt: float, checkpoint=False):
    global best_acc
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, (inputs, targets, _) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = subsample_inputs(inputs, stride)
            outputs = model(inputs, scale_dt=scale_dt)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: (%d/%d) | Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (batch_idx, len(dataloader), eval_loss/(batch_idx+1), 100.*correct/total, correct, total)
            )

    # Save checkpoint.
    if checkpoint:
        acc = 100.*correct/total
        if acc > best_acc:
            state = {
                'model': model.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            best_acc = acc

        return acc

pbar = tqdm(range(start_epoch, args.epochs))
val_acc_epochs = []
train_acc_epochs = []
stride_epochs = []
scale_dt_epochs = []
cumulative_times = []  # seconds since start_epoch loop began
t0 = time.time()
for epoch in pbar:
    if epoch == 0:
        pbar.set_description('Epoch: %d' % (epoch))
    else:
        pbar.set_description('Epoch: %d | Val acc: %1.3f' % (epoch, val_acc))
    stride, scale_dt = curriculum(epoch)
    stride_epochs.append(stride)
    scale_dt_epochs.append(scale_dt)
    train_acc = train(stride, scale_dt)
    val_acc = eval(epoch, valloader, stride, scale_dt, checkpoint=True)
    val_acc_epochs.append(val_acc)
    train_acc_epochs.append(train_acc)
    eval(epoch, testloader, stride, scale_dt)
    # scheduler.step()
    # print(f"Epoch {epoch} learning rate: {scheduler.get_last_lr()}")
    # ---- timing ----
    if device == "cuda":
        torch.cuda.synchronize()  # ensures all queued GPU work is finished before timing
    cumulative_times.append(time.time() - t0)

print(val_acc_epochs)
print(train_acc_epochs)
print(cumulative_times)

if args.val_acc_dir != "":
    os.makedirs(os.path.dirname(args.val_acc_dir), exist_ok=True) if os.path.dirname(args.val_acc_dir) else None

    # Save both in one .npz file (recommended)
    # If user passes ".../val_acc.npy", we will still write ".../val_acc.npz".
    base, ext = os.path.splitext(args.val_acc_dir)
    save_path = base + ".npz"

    np.savez(
        save_path,
        val_acc_epochs=np.array(val_acc_epochs, dtype=np.float32),
        train_acc_epochs=np.array(train_acc_epochs, dtype=np.float32),
        cumulative_time_sec=np.array(cumulative_times, dtype=np.float64),
        stride_epochs=np.array(stride_epochs, dtype=np.int32),
        scale_dt_epochs=np.array(scale_dt_epochs, dtype=np.float32),
    )
    print(f"Saved val_acc_epochs and cumulative_time_sec to {save_path}")