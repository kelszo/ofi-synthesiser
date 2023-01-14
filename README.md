# ofi-synthesiser

## Project Structure

```
├── conf                # Configuration files
├── data                # Data files
│   ├── external        # External data
│   ├── interim         # Intermediate processed data
│   ├── processed       # Processed data
│   └── raw             # Raw data (read only)
├── ofisynthesiser      # Main project source code
│   ├── data            # Code that handles data processing
│   ├── executors       # Scripts that run code end-to-end
│   ├── models
│   ├── networks        # Interchangeable networks that the models rely on
│   └── utils           # Utils than can be used in several places
├── local               # A non-pushed dir for local work
├── models              # Compiled models
├── notebooks           # Notebooks used for literate programming and executors
├── references          # Manuals and other material
├── report
└── results             # Results from training

```

## Setup
To create the env: `make conda_create`

To activate the env: 