# Continuous Sequential Models

This is the code repository accompanied the manuscript "Continuous Laws for Sequential Models."

## Refining the Mesh in State Space Models

Code scripts in the directory [refined_mesh](./refined_mesh/) are used to produce the results in Figure 4, where a pretrained S4 model and a pretrained S6 model are evaluated under temporal refinement.

**IMPORTANT:** The training and evaluation scripts import the S4 and S6 modules with scale_dt integrated. The model scripts can be found in [src/s4d.py](src/s4d.py) and [src/mamba.py](src/mamba.py), respectively. Depending on where you put these two scripts in your system, you may have to change the import lines in the training and evaluation scripts. To switch from the default ZOH discretization to the bilinear discretization, pass in `dis_mode="bilinear"`.

### Data Generation

To generate the dynamical system training data, one can run the data generation scripts:

* Van der Pol Oscillator: 

```
python data_gen/generate_VDP.py \
  --out data/vdp_mu.npz \
  --L 2048 \
  --delta 0.005 \
  --n_train 0 --n_val 2000 --n_test 2000 \
  --mu_min 0.2 --mu_max 2.0 \
  --x0_range 1 1 --v0_range 0 0 \
  --burn_in 4000 \
  --noise_std 0.00
```

* Forced Duffing:

```
python data_gen/generate_FD.py \
  --out data/fd_gamma.npz \
  --L 1024 --delta 0.01 \
  --alpha -1.0 --beta 1.0 --damp 0.10 --omega 1.2 \
  --gamma_min 0.4 --gamma_max 0.8 \
  --burn_in 0 --extra 0 --normalize
```

* Damped Harmonic Oscillator:

```
python data_gen/generate_DHO.py \
  --out data/dho_omega_zeta.npz \
  --L 1024 \
  --delta 0.01 \
  --n_train 20000 --n_val 2000 --n_test 2000 \
  --omega_min 0.5 --omega_max 2.0 \
  --zeta_min 0.05 --zeta_max 0.2 \
  --x0_range 1 1 --v0_range 0 0 \
  --noise_std 0.01 --normalize
```

* Ornstein--Uhlenbeck Process

```
python data_gen/generate_OU.py \
  --out data/ou_theta_sigma.npz \
  --L 1024   --delta 0.01 \
  --n_train 20000 --n_val 2000 --n_test 2000 \
  --theta_min 0.5 --theta_max 2.0 \
  --sigma_min 0.01 --sigma_max 0.01  \
  --x0_mode uniform   --burn_in 0   --meas_noise_std 0.0
```

### Training

The training scripts can be found in [refined_mesh/training](./refined_mesh/training/). One just needs to pass in the dataset generated using the generation scripts via the ``--data`` flag.

### Evaluation

To evaluate the model with different a $\tau$, one needs to call the scripts in [refined_mesh/evaluation](./refined_mesh/evaluation/) and pass in the desired $\tau$ as `--scale_dt`. For example:

```
python eval_DS_s4.py --data data/dho.npz --batch_size 128 --ckpt_dir checkpoint_dho --scale_dt 1
```

and

```
python eval_DS_mamba.py --data data/dho.npz --batch_size 128 --ckpt_dir checkpoint_dho --scale_dt 1
```
Note that to change the discretization step size, you will have to regenerate the test data using the data generation script. In particular, set `--delta` to be `0.01 * scale_dt` and `--L` to be `int(1024 / scale_dt)`.

## Training Models on Quantized Dynamical Systems Data

Code scripts in the directory [ds_forecasting](./ds_forecasting/) are used to produce the results in Figure 5 (Left).

### Data Generation

To generate data, run the scripts in [data_gen](./ds_forecasting/data_gen/). For example:
```
python generate_VDP_forecasting.py  --out data/fd_0  --eta 0
python generate_FD_forecasting.py  --out data/fd_0  --eta 0
python generate_DHO_forecasting.py  --out data/fd_0  --eta 0
python generate_OU_forecasting.py  --out data/fd_0  --eta 0
```
To change the continuity level $\eta$, pass it into the generation script using `--eta`.

### Training

S4, S6, and Transformers can be trained using the training scripts `train_S4.py`, `train_S6.py`, and `train_transformers.py`, respectively, by passing in the data you generated using the data generation scripts. To train other models (DeltaNet, Linear Attention, Mamba-3, etc.), use the training script `train_rest.py` and pass in the desired models class via `--model`.

## Training Subsampled Models on Long-Range Arena

Code scripts in the directory [subsampled_training](./subsampled_training/) are used to produce the results in Figure 6.

The training is heavily based on the [s4](https://github.com/state-spaces/s4) repository. To implement the pretraining scripts for subsampled S4D model, you will have to replace the files [block.py](http://github.com/state-spaces/s4/blob/main/src/models/sequence/backbones/block.py), [model.py](http://github.com/state-spaces/s4/blob/main/src/models/sequence/backbones/model.py), and [ssm.py](https://github.com/state-spaces/s4/blob/main/src/models/sequence/kernels/ssm.py) found in [s4](https://github.com/state-spaces/s4) with the corresponding files in [src/subsampled_models/](./src/subsampled_models/) in this repository.

The full pretraining script can be found at [subsampled_training/train_subsample.py](subsampled_training/train_subsample.py). This plays exactly the same role as [training.py](https://github.com/state-spaces/s4/blob/main/train.py) in the [s4 repository](https://github.com/state-spaces/s4).

For illustration purposes, we also provide a demo script in [subsampled_training/demo/](subsampled_training/demo/). This script can be called directly to train a subsampled S4D model on PathX.