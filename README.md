# ofi-synthesiser
[![License: AGPL-3.0](https://img.shields.io/github/license/kelszo/ofi-synthesiser)](https://opensource.org/licenses/AGPL-3.0)

Thesis project generating synthetic data for the Swedish Trauma Registry (SweTrau) with the goal of improving the prediction performance for identifying opportunities for improvement (OFIs) in trauma patient care using machine learning models.

## Project Structure

```
├── conf                # Configuration files
├── data                # Data files
│   ├── external        # External data
│   ├── interim         # Intermediate processed data
│   ├── processed       # Processed data
│   └── raw             # Raw data (read only)
├── local               # A non-pushed dir for local work
├── models              # Compiled models
├── notebooks           # Notebooks used for literate programming and executors
├── ofisynthesiser      # Main project source code
│   ├── data            # Code that handles data processing
│   ├── executors       # Scripts that run code end-to-end
│   ├── models
│   ├── networks        # Interchangeable networks that the models rely on
│   └── utils           # Utils than can be used in several places
├── out                 # Safe space for output
├── report
└── results             # Results from training

```

## Setup
Use `$ nix-shell` to enter the dev environment. Alternatively build the docker (podman) image with: `$ docker_build && docker_load`

To run executors use: `$ python -m <dir.to.file>` (ex. `$ python -m ofisynthesiser.executors.run`)

## License
[AGPL-3.0](https://opensource.org/licenses/AGPL-3.0), see `LICENSE`