# `hh4b` Analysis with SPANet
In this analysis, we apply the Symmetry Preserving Attention Network ([SPANet](https://github.com/Alexanders101/SPANet)) on the `hh4b` dataset, considering both resonant and non-resonant scenarios.

## Installation
Install this repository in your SPANet directory.
```bash
cd <path/to/SPANet>
git clone https://github.com/r10222035/hh4b_SPANet
````

## Environment
Set up the environment following the SPANet [documentation](https://github.com/Alexanders101/SPANet?tab=readme-ov-file#installation) to install the required dependencies.


## Dataset
The training and testing datasets can be obtained from [Zenodo](https://zenodo.org/records/10952296). Use the following commands to download the datasets:

For Resonant Dataset:

```bash
wget -O ./hh4b_SPANet/data/hh4b_resonant_train.h5 https://zenodo.org/records/10952296/files/hh4b_resonant_train.h5
wget -O ./hh4b_SPANet/data/hh4b_resonant_test.h5 https://zenodo.org/records/10952296/files/hh4b_resonant_test.h5
```

For Non-Resonant Dataset:
```bash
wget -O ./hh4b_SPANet/data/hh4b_nonresonant_train.h5 https://zenodo.org/records/10952296/files/hh4b_nonresonant_train.h5
wget -O ./hh4b_SPANet/data/hh4b_nonresonant_test.h5 https://zenodo.org/records/10952296/files/hh4b_nonresonant_test.h5
````

## Training

To train SPANet on the `hh4b` dataset, execute the following commands:

For Resonant Dataset:
```bash
python -m spanet.train -of hh4b_SPANet/options_files/hh4b_resonant.json --log_dir spanet_output  --name hh4b_resonant
```

For Non-Resonant Dataset:
```bash
python -m spanet.train -of hh4b_SPANet/options_files/hh4b_nonresonant.json --log_dir spanet_output  --name hh4b_nonresonant
```

The `--log_dir` is the output directory. The `--name` is the output sub-directory. The training results are saved in `spanet_output/hh4b_resonant/` or `spanet_output/hh4b_nonresonant/` for the above commands.

## Testing

To evaluate the performance of the trained model on the testing dataset, use the following commands:

For Resonant Dataset:
```bash
python -m spanet.test ./spanet_output/hh4b_resonant/version_0 -tf ./hh4b_SPANet/data/hh4b_resonant_test.h5 --gpu
```

For Non-Resonant Dataset:
```bash
python -m spanet.test ./spanet_output/hh4b_nonresonant/version_0 -tf ./hh4b_SPANet/data/hh4b_nonresonant_test.h5 --gpu
```

The `--gpu` flag is optional. If you trained many times, then you should adjust the version number accordingly.

## Prediction

To generate predictions on a set of events for further analysis, run the following commands:

For Resonant Dataset:
```bash
python -m spanet.predict ./spanet_output/hh4b_resonant/version_0 ./spanet_hh4b_resonant_output.h5 -tf ./hh4b_SPANet/data/hh4b_resonant_test.h5 --gpu --output_vectors
```

For Non-Resonant Dataset:
```bash
python -m spanet.predict ./spanet_output/hh4b_nonresonant/version_0 ./spanet_hh4b_nonresonant_output -tf ./hh4b_SPANet/data/hh4b_nonresonant_test.h5 --gpu --output_vectors
```

The `--output_vectors` is optional and is used to output the embedding vectors.

This will create an HDF5 file with the same structure as the testing dataset. The jet labels and the class label would be replaced by the SPANet predictions. If the `--output_vectors` is used, SPANet would add embedding vectors in the HDF5 file.

## Event `.yaml` File
We will go over the `hh4b` event configuration.

Following is the event description for a resonant `hh4b` event.
This file is located at [`event_files/hh4b_resonant.yaml`](./event_files/hh4b_resonant.yaml).

```yaml
# ---------------------------------------------------
# REQUIRED - INPUTS - List all inputs to SPANet here.
# ---------------------------------------------------
INPUTS:
  # -----------------------------------------------------------------------------
  # REQUIRED - SEQUENTIAL - inputs which can have an arbitrary number of vectors.
  # -----------------------------------------------------------------------------
  SEQUENTIAL:
    Source:
      mass: log_normalize
      pt: log_normalize
      eta: normalize
      phi: normalize
      btag: none

  # ---------------------------------------------------------------------
  # REQUIRED - GLOBAL - inputs which will have a single vector per event.
  # ---------------------------------------------------------------------
  GLOBAL:


# ----------------------------------------------------------------------
# REQUIRED - EVENT - Complete list of resonance particles and daughters.
# ----------------------------------------------------------------------
EVENT:
  h1:
    - b1
    - b2
  h2:
    - b1
    - b2

# ---------------------------------------------------------
# REQUIRED KEY - PERMUTATIONS - List of valid permutations.
# ---------------------------------------------------------
PERMUTATIONS:
    EVENT:
      - [ h1, h2 ]
    h1:
      - [ b1, b2 ]
    h2:
      - [ b1, b2 ]


# ------------------------------------------------------------------------------
# REQUIRED - REGRESSIONS - List of desired features to regress from observables.
# ------------------------------------------------------------------------------
REGRESSIONS:


# ------------------------------------------------------------------------------
# REQUIRED - CLASSIFICATIONS - List of desired features to regress from observables.
# ------------------------------------------------------------------------------
CLASSIFICATIONS:
  EVENT:
    - signal
```

### `INPUTS`
The Inputs section of the `.yaml` file defines which features are present in datasets. SPANet will use these features to make predictions.

```yaml
SEQUENTIAL:
    Source:
```
This defines a sequential (variable length) input named `Source`. We define five features.
- `mass`
- `pt`
- `eta`
- `phi`
- `btag`

We store the 4-vector and the b-tag for each jet. Notice that we `log_normalize` the `mass` and `pt` features, normalize the `eta` and `phi` features, and don't do anything with `btag` because it is already binary-valued.

### `EVENT`
This defines the event structure.

We have two Higgs bosons in an event denoted by `h1` and `h2`. Each Higgs boson decays into the bottom and anti-bottom quark pair denoted by `b1` and `b2`.

### `PERMUTATIONS`
This defines invariant symmetries for the jet assignment task.

We tell SPANet that the particular ordering of `h1` and `h2` doesn't matter. The bottom and anti-bottom are indistinguishable in the detector. Therefore, we set `[b1, b2]` to be a valid permutation. 

### `REGRESSIONS`
This section defines event, particle, and decay product regressions. This is not used for `hh4b` analysis.
 
### `CLASSIFICATIONS`
This section defines event, particle, and decay product classifications.

We define an event-level classification to distinguish the signal and background events.


## Options File

### Combined Training
We consider jet assignment and classification tasks in the `hh4b` analysis. These outputs will be trained simultaneously. The strength of each target is controlled by following hyperparameters.
```json
// From `options_files/hh4b_resonant.json`
"assignment_loss_scale": 1.0,
"classification_loss_scale": 1.0,
```
These will control the absolute weight assigned to each loss term in the total loss function.

## `hh4b` Dataset Structure
We demonstrate the resonant dataset structure.

You can examine the HDF5 file structure with the following command:
```bash
$ python utils/examine_hdf5.py hh4b_SPANet/data/hh4b_resonant_test.h5  --shape

============================================================
| Structure for hh4b_SPANet/data/hh4b_resonant_test.h5
============================================================

|-CLASSIFICATIONS
|---EVENT
|-----signal                     :: int64    : (100000,)
|-INPUTS
|---Source
|-----MASK                       :: bool     : (100000, 10)
|-----btag                       :: bool     : (100000, 10)
|-----eta                        :: float32  : (100000, 10)
|-----mass                       :: float32  : (100000, 10)
|-----phi                        :: float32  : (100000, 10)
|-----pt                         :: float32  : (100000, 10)
|-TARGETS
|---h1
|-----b1                         :: int64    : (100000,)
|-----b2                         :: int64    : (100000,)
|---h2
|-----b1                         :: int64    : (100000,)
|-----b2                         :: int64    : (100000,)
```