# [Agriculture Data Analytics](./../../../)

# Install [nbautoexport](https://github.com/drivendataorg/nbautoexport)

 nbautoexport allows you to automatically export Jupyter notebooks to other formats, including PY files.  It is a Jupyter extension.  We use it export our notebooks to Python files to assist Pull Requests.

## Installation

First, you will need to install `nbautoexport`. This should be installed in the same environment you are running Jupyter Notebook or Jupyter Lab from. `nbautoexport` is available either from [PyPI](https://pypi.org/project/nbautoexport/) via `pip` or from [conda-forge](https://github.com/conda-forge/nbautoexport-feedstock) via `conda`.

Then, to register `nbautoexport` to run automatically while using Jupyter Notebook or Jupyter Lab, run:

```bash
conda install nbautoexport --channel conda-forge
```

Then, to register `nbautoexport` to run automatically while using Jupyter Notebook or Jupyter Lab, run:

```bash
nbautoexport install
```

If you already have a Jupyter server running, you will need to restart it for this to take effect.

# Simple usage

Let's say you have a directory for your project and you keep your notebooks in a `population-planning-data-analytics` directory at
`C:\Users\USER\Documents\GitHub\population-planning-data-analytics\`.

To configure that directory for automatic exporting, open a command prompt at `C:\Users\USER\Documents\GitHub\` run the following:

```bash
nbautoexport configure population-planning-data-analytics
```

This will create a configuration file `C:\Users\USER\Documents\GitHub\population-planning-data-analytics\.nbautoexport`.  

## Sub directories

If you have Jupyter notebooks in subdirectories, lets say `samples` you need to open the Anaconda Prompt (Anaconda3).

Change directory to the Repository Directory root\ eg:

```bash
cd C:\Users\USER\Documents\GitHub\population-planning-data-analytics  
```

and then run  
```bash
nbautoexport configure samples  
```

This will create a configuration file `C:\Users\USER\Documents\GitHub\population-planning-data-analytics\samples\.nbautoexport`.  

***Note:  My directory path will be different to your directory path***

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.