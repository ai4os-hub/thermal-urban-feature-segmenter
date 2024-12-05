# thermal-urban-feature-segmenter

[![Build Status](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/thermal-urban-feature-segmenter/main)](https://jenkins.services.ai4os.eu/buildStatus/icon?job=AI4OS-hub/thermal-urban-feature-segmenter/main)

Deepaas API for thermal urban feature segmentation (TUFSeg). This code makes use of the [TUFSeg model repo](https://github.com/emvollmer/TUFSeg).

To facilitate setting up, the bash script `setting_up_deployment.sh` can be 
run to install everything automatically:
```bash
wget https://raw.githubusercontent.com/ai4os-hub/thermal-urban-feature-segmenter/master/setting_up_deployment.sh
source setting_up_deployment.sh
```
This takes care of all required installations and finishes by running [deepaas](https://github.com/ai4os/DEEPaaS).

Alternatively, simply make use of the Docker image, buildable via the Dockerfile.

## Project structure

```
├── Jenkinsfile             <- Describes basic Jenkins CI/CD pipeline
├── LICENSE                 <- License file
├── README.md               <- The top-level README for developers using this project.
├── VERSION                 <- Version file indicating the version of the model
│
├── tufsegm_api
│   ├── README.md           <- Instructions on model integration with DEEPaaS.
│   ├── __init__.py         <- Makes tufsegm_api a Python module, contains main functions
│   ├── utils.py            <- Helper functions
│   └── config.py           <- Module to define CONSTANTS used across the AI-model python package
│
├── api                     <- API subpackage for the integration with DEEP API
│   ├── __init__.py         <- Makes api a Python module, includes API interface methods
│   ├── config.py           <- API module for loading configuration from environment
│   ├── responses.py        <- API module with parsers for method responses
│   ├── schemas.py          <- API module with definition of method arguments
│   └── utils.py            <- API module with utility functions
│
├── data                    <- Data subpackage for the integration with DEEP API
│
├── docs                   <- A default Sphinx project; see sphinx-doc.org for details
│
├── models                 <- Folder to which models are automatically stored 
|                             (if not on nextcloud)
│
├── notebooks              <- Jupyter notebooks
│
├── references             <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting
│
├── requirements-dev.txt    <- Requirements file to install development tools
├── requirements-test.txt   <- Requirements file to install testing tools
├── requirements.txt        <- Requirements file to run the API and models
│
├── pyproject.toml         <- Makes project pip installable (pip install -e .)
│
├── tests                   <- Scripts to perform code testing
│   ├── configurations      <- Folder to store the configuration files for DEEPaaS server
│   ├── conftest.py         <- Pytest configuration file (Not to be modified in principle)
│   ├── data                <- Folder to store the data for testing
│   ├── models              <- Folder to store the models for testing
│   ├── test_deepaas.py     <- Test file for DEEPaaS API server requirements (Start, etc.)
│   ├── test_metadata       <- Tests folder for model metadata requirements
│   ├── test_predictions    <- Tests folder for model predictions requirements
│   └── test_training       <- Tests folder for model training requirements
│
└── tox.ini                <- tox file with settings for running tox; see tox.testrun.org
```

## Model Integration with DEEPaaS

The folder `tufsegm_api` is designed to contain code with which to access the model code from the submodule [TUFSeg](https://github.com/emvollmer/TUFSeg).
The folder `tufsegm_api` contains an `__init__.py` file conserving the already defined methods.

Methods in `tufsegm_api` are used by the subpackage `api` to define the API interface.
See the project structure section for more information about the `api` folder.
API and CLI arguments and responses are adapted in `api.schemas` and `api.responses`.

## Testing

Testing process is automated by tox library. 
Tests are implemented following [pytest](https://docs.pytest.org) framework.
Fixtures and parametrization are placed inside `conftest.py` files while
assertion tests are located on `test_*.py` files.

    - tests/data: Contains testing data (sample images).
    - tests/models: Contains a dummy model.
    - tests/test_metadata: Tests for get_metadata functionality.
    - tests/test_predictions: Tests for inference functionality.
    - tests/test_training: Tests for training functionality.

The dummy model is of the smallest possible size, but still comparatively
large. For this reason, the tests take a bit of time to run.

Running the tests with tox:

```bash
$ pip install -r requirements-dev.txt
$ tox
```

Running the tests with pytest:

```bash
$ pip install -r requirements-test.txt
$ python -m pytest --numprocesses=auto --dist=loadscope tests
```
or for more detailled information
```bash
$ python -m pytest tests
```
