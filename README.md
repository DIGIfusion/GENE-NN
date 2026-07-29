# GENE-NN

Neural-network surrogate models for predicting local, linear gyrokinetic outputs from GENE, a gyrokinetic plasma turbulence code.

This repository contains the trained models and inference tools needed to apply models to new inputs. Inference can currently be run from:

- CSV datasets
- EQDSK + ITERDB profile files

The main application is a proof-of-principle surrogate for MAST-U pedestal linear GENE simulations.


## Supported model types

Two saved model types are supported:

1. **Multi-head MLP regression (MH-MLP)**

   A direct multi-output neural network that predicts all target quantities from the input features.

2. **Class-regression multi-head model (MH Class-Reg)**

   A two-stage model that first classifies `omega` into predefined regimes, then applies a class-specific multi-head regressor.


## Input features

The MAST-U models expect the following input features:

| Feature | Description |
|---|---|
| `Bref` | Reference magnetic field used in the GENE normalization, $B_\mathrm{ref}$ |
| `Lref` | Reference length used in the GENE normalization, $L_\mathrm{ref}$ |
| `beta` | Local normalized plasma beta, $\beta$ |
| `coll` | Collisionality, $\nu$ |
| `omn_e` | Electron density gradient, $a/L_{n_e}$ |
| `omt_e` | Electron temperature gradient, $a/L_{T_e}$ |
| `q0` | Safety factor at the local flux surface, $q$ |
| `shat` | Magnetic shear at the local flux surface, $\hat{s}$ |
| `dpdx_pm` | Ratio of the negative equilibrium pressure gradient and the reference magnetic pressure, $-\nabla p / p_m$, where $p_m = B_\mathrm{ref}^2/(2\mu_0)$  |
| `omegatorref` | Toroidal rotation at the local flux surface, $\Omega_\mathrm{tor,ref}$ |
| `kymin` | Binormal wavenumber used for the local linear GENE run, $k_y \rho_s$ |

For CSV inference, these columns must already be provided in the same convention as GENE inputs. For EQDSK+ITERDB inference, the script calculates these quantities automatically from the input files.

## Predicted targets

The multi-head MLP model predicts:


| Target | Description |
|---|---|
| `gamma` | Linear growth rate, $\gamma$ |
| `omega` | Real mode frequency, $\omega$  |
| `Chi_i/Chi_e` | Ratio of ion to electron heat diffusivity, $\chi_i / \chi_e$ |
| `D_e/Chi_e` | Ratio of electron particle diffusivity to electron heat diffusivity, $D_e / \chi_e$ |

The classification-regression model predicts `omega`, `Chi_i/Chi_e`, and `D_e/Chi_e`.

The output CSV contains the original input columns, followed by prediction columns named for example:

```text
pred_gamma
pred_omega
pred_Chi_i/Chi_e
pred_D_e/Chi_e
```
The predicted outputs use the same convention as GENE. In particular, `gamma` and `omega` are the GENE-normalized linear growth rate and real frequency, normalized by $c_\mathrm{ref}/L_\mathrm{ref}$. The ratio targets `Chi_i/Chi_e` and `D_e/Chi_e` are dimensionless.

For class-regression models, the output also includes the predicted class and class probabilities.

## Saved model format

The model type is detected automatically from the saved model directory.

### Multi-head MLP

```text
model/MAST_U/MH_MLP/
  model.pt
  model_config.json
  preprocessor.joblib
  target_scaler.joblib
```

### Class-regression model

```text
model/MAST_U/MH_CLASSREG/
  pipeline_config.json
  classifier/
    model.pt
    model_config.json
    preprocessor.joblib
  regressors/
    class_0/
    class_1/
    class_2/
```

Each regressor class directory contains its own saved model, configuration, input preprocessor, and target scaler.


## CSV inference

Run inference on an existing tabular dataset:

```bash
python -m gene_nn.cli.inference_from_dataset \
  --csv path/to/input.csv \
  --model-dir path/to/model/MAST_U/MH_MLP \
  --out-csv path/to/predictions.csv
```

or, for the class-regression model:

```bash
python -m gene_nn.cli.inference_from_dataset \
  --csv path/to/input.csv \
  --model-dir path/to/model/MAST_U/MH_CLASSREG \
  --out-csv path/to/predictions.csv
```
For CSV inference, the input columns should be provided in the same format and normalization as used in the GENE input files.


## EQDSK + ITERDB inference

Run inference directly from EQDSK and ITERDB files:

```bash
python -m gene_nn.cli.inference_from_eqdsk_iterdb \
  --eqdsk-path path/to/EQDSK_COCOS_02.OUT \
  --iterdb-path path/to/iterdb \
  --model-dir path/to/model/MAST_U/MH_CLASSREG \
  --x0 0.95 0.96 0.97 0.98 \
  --kymin 0.1 0.2 0.3 0.4 0.5 \
  --out-csv path/to/predictions.csv
```

The radial locations `--x0` must be provided explicitly.

If `--kymin` is not provided, the default values are:

```text
0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
```

For EQDSK/ITERDB inference, the required model inputs are calculated automatically from the equilibrium and profile files before running the surrogate model.


## Parameter overrides and scans

The EQDSK/ITERDB inference script supports fixed overrides and parameter scans.

### Fixed overrides

```bash
--set beta=0.02 omegatorref=0.0
```

This replaces the generated values with fixed values before inference.

### Absolute scans

```bash
--scan beta=0.001,0.003,0.005
```

This creates separate rows using exactly the listed values. The original generated value is not added automatically.

### Multiplicative scans

```bash
--scale-scan beta=0.9,1.0,1.1
```

This multiplies the generated value by each listed factor. Use factor `1.0` to include the unscaled base case.

Multiple scan parameters are expanded as a Cartesian product.

## Installation

From the repository root, install the package and its dependencies with:

```bash
python -m pip install -e .
```

The package requires Python 3.10 or newer. The main runtime dependencies are `numpy`, `pandas`, `scipy`, `scikit-learn`, `joblib`, and `torch`.

## Notes and limitations

- The current models are trained on ion-scale local linear GENE simulations.
- The current MAST-U models are trained in the parameter space of a single discharge and should not be assumed to generalize outside that domain without validation.
- `x0` is the normalized toroidal flux coordinate, `rho_tor`.
- Very close to the separatrix, the magnetic shear $(\hat{s})$ values calculated from EQDSK/ITERDB may slightly differ from values calculated internally by GENE, so near-edge `shat` should be checked when needed.
