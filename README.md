# enrichIFC.

## Installation

In order to run this repository, Anaconda is to be installed on your machine.

### Create new Anaconda environment

open Anaconda console
```bash
conda create --name dev-ifc python=3.11
```

### Activate newly created environment
```bash
conda activate dev-ifc
```

### export the conda environment settings to .yml 
```bash
conda env export > ifc-dev.yml
```

### (re)create the conda environment from a .yml
```bash
conda env create -f ifc-dev.yml
```

### update the existing conda environment from a .yml (NOT recommended)
```bash
conda env update --file ifc-dev.yml --prune
```

### Install required Dependencies

- For plotting svgs, chromedriver-binary is needed `pip install chromedriver-binary-auto`(https://pypi.org/project/chromedriver-binary-auto/)
- Change the working directory to the project path  `cd /D ... `
- Inside the anaconda console and the activated env, run the command `pip install -r requirements.txt`
- readme test.