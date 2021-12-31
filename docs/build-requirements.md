# [Data Analytics Project Template](./../../../)

# Build requirements.txt

A requirements.txt file is used to specify the build requirements for a project or a Jupyter notebook.  To ensure when anyone opens a Jupyter notebook, they have the latest version of the dependencies, we need to build the notebook `requirements.txt`.  Each Jupyter notebook should have a cell that runs the following code:

```python
# Local
!pip install -r script/requirements.txt --quiet

# Remote option
!pip install -r https://github.com/markcrowe-com/data-analytics-project-template/blob/master/notebooks/script/requirements.txt?raw=true  --quiet
```

## Install Tools

To build a requirements.txt file for a Jupyter notebook first install [nbautoexport](install-nbautoexport.md)

Next install pipreqs  
```bash
pip install pipreqs
```
## Build the requirements.txt file
To build the `requirements.txt` file for your notebooks go the associated nbautoexport script folder and run the following command:

```bash
pipreqs
```

### Example

Let's say you have a project and keep your notebooks in a `notebooks` directory at
`C:\Users\USER\Documents\GitHub\data-analytics-project-template\notebooks`.
You need to open the Anaconda Prompt and run the following command 

```bash
cd C:\Users\USER\Documents\GitHub\data-analytics-project-template\notebooks\script

pipreqs
```

A `requirements.txt` file will be created at `C:\Users\USER\Documents\GitHub\data-analytics-project-template\notebooks\script\requirements.txt`.


The in the jupyter notebook add a cell with the following command to install the dependencies:

```python
# Remote option
!pip install -r https://github.com/markcrowe-com/data-analytics-project-template/blob/master/notebooks/script/requirements.txt?raw=true  --quiet

# Local Option
#!pip install -r script/requirements.txt --quiet

```

***Note: My directory path will be different to your directory path***

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.