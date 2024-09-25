# signature_survival
Repository for the paper "Dynamical Survival Analysis with Controlled Latent States". 

## Installation
Clone the repository, then inside the folder, use a `virtualenv` to install the requirements
```shell script
git clone https://github.com/LinusBleistein/signature_survival.git
cd signature_survival

# At the moment, we support the Python 3.6 or lower version
# If your default interpreter is Python3:
virtualenv .venv_SigSurv
# If your default interpreter is Python2, you can explicitly target Python3 with:
virtualenv -p python3.6 .venv_SigSurv

source .venv_SigSurv/bin/activate
```
Then, to download all required modules and initialize the project run the following commands:
```shell script
pip install -r requirements.txt
pip install -e .
```
The second command installs the project as a package, making the main module importable from anywhere.

Then, to add the virtual eviroment to jupyter kernel:
```shell script
python -m ipykernel install --user --name=".venv_SigSurv"
```


## Reproducing the experiments
To reproduce the results used in the paper, simply run 
```shell script
python ./experiment/experiment.py
```