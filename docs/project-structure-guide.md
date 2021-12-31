# [Data Analytics Project Template](./../../../)

## Project Structure Guide
> "I bring order to chaos" - The Borg Queen, 2373

The project structure guide defines the layout of the project and the files and folders that make up the project.  

### Directory Structure
```txt
📦data-analytics-project-template | The project root folder
 ┣ 📂.vscode                      | VS Code generated content: settings for spellings.
 ┣ 📂artifacts                    | Work products of the Jupyter notebooks.
 ┣ 📂assets                       | Assets for the project, provided by the starter pack.
 ┣ 📂docs                         | Documentation for the project.
 ┃ ┗ 📂images                     | Images for the documentation.
 ┣ 📂junk-dna                     | Code and artifacts that didn't make it into the release.   
 ┣ 📂maps                         | Map system files.
 ┃ ┗ 📂irl-adm1                   | irl-adm1 maps files.
 ┣ 📂notebooks                    | Jupyter notebooks for the project.
 ┃ ┗ 📂script                     | nbautoexport folder: Its contents are auto generated.
 ┣ 📂python-package               | Python source code for the Python package.
 ┃ ┗ 📂project_name               | project_name Python package.
 ┣ 📂references                   | References for the project, with copies.
 ┗ 📜readme.md                    | The project readme file.
```
> Directory Structure Legend

### Naming Conventions

#### File and Directory Naming Conventions 
All directories, Jupyter notebooks, markdown files, images and other files are to be named in lower kebab-case. This rule does not apply to Python files in the [python-package](./../python-package/) folder or to map system files in the sub-directories of [maps](./../maps/).

Lower kebab-case is chosen because the project is available online as a data science portfolio project on GitHub and GitHub URLs are case sensitive.

##### Example: Same URL except for case. (1) readme, (2) README.
https://github.com/markcrowe-com/
1.  <https://raw.githubusercontent.com/markcrowe-com/data-analytics-project-template/master/readme.md>  
  Returns the contents of `readme.md`

2.  <https://raw.githubusercontent.com/markcrowe-com/data-analytics-project-template/master/README.md>  
   Returns a `404: Not Found` error

##### Correct file name examples:

| filename |
|---------------------------|
| `readme.md` |
| `docs/assessment-criteria.md` |
| `docs/images/gantt-chart.png` |
| `assets/births-deaths-marriages-ireland-1960-2021` |
| `notebooks/notebook-1-01-example.ipynb` |


##### Incorrect file name example:

| filename | Reason |
|---------------------------|---------------------------------------------------|
| `README.md` | Capitalized filename |
| `assets/birthsdeathsmarriagesireland-1960-2021` | Difficult to read, words not separated with '-' |
| `Docs/Assessment-Criteria.md` | Mixed case filename, Capitalized First letters in words is PascalCase |
| `DOCS/` | Capitalized directory |
| `docs/images/gantt-chart.PNG` | Capitalized file extension |
| `notebooks/notebook-1-01 example.ipynb` | Space in name |
| `samples/samplePythonModule.py` | Mixed case filename, Python filename words not separated with '_' |

#### Jupyter Notebooks
Each notebook filename should begin `notebook-S-NN-` where S is a unique number and NN is a unique two-digit number of the notebook. This code is used to order the notebooks in the project.

#### Python File Naming Conventions

Python files in the [python-package](./../python-package/) folder are to be named in snake_case.

### Directory Notes
When working on this project use the following directory structure:

#### artifacts
Files generated from the Jupyter notebooks should be placed in [artifacts](./../artifacts/) directory.

#### assets
The files provided in the [assets](./../assets/) directory were provided to start the project. They are not to be modified.

#### docs
Documentation for the project is to be placed in the [docs](./../docs/) directory. Documentation is to be where possible in markdown

##### docs/images
Images used in the documentation should be placed in the [docs/images](./../docs/images/) directory.

#### maps
Map system files for the project should be placed in the [maps](./../maps/) directory. Each map's files should be in it's own sub directory. These files names are not to be changed to conform to filename conventions.

#### notebooks
Jupyter notebooks for the project should be placed in the [notebooks](./../notebooks/) directory. Each notebook filename should begin `notebook-S-NN-` where S is a unique number and NN is a unique two-digit number of the notebook. It may be necessary to clear/delete files from the [notebooks/script](./../notebooks/script/) directory in teh event a notebook is renamed or deleted.

#### python-package
Python modules files for the `project_name` Python package are kept in the [python-package](./../python-package/) directory. These files and folders folder are to be named in snake_case.

#### references
References are to be listed using Harvard referencing style in the file [/references/readme.md](./../references/readme.md). Copies of the references are to be placed in the [references](./../references/) directory beginning with their data accessed in the filename e.g. `2021-11nov-04-python-3-f-strings.md`. Where possible reference copies are to be saved as markdown or pdf.

### Jupyter Notebooks Online

To make the project interactive online each Jupyter Notebook will include a heading with the online editors in the first cell.

<table><tr><td><a href="https://mybinder.org/v2/gh/markcrowe-com/data-analytics-project-template/master?filepath=notebooks/notebook-1-01-example-bad-code-population.ipynb" target="_parent"><img src="https://mybinder.org/badge_logo.svg" alt="Open In Binder"/></a></td><td>online editors</td><td><a href="https://colab.research.google.com/github/markcrowe-com/data-analytics-project-template/blob/master/notebooks/notebook-1-01-example-bad-code-population.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></td></tr></table>

Two function `create_jupyter_notebook_header(github_username: str, repository: str, notebook_filepath: str, branch: str)` and `print_jupyter_notebook_header_html(github_username: str, repository: str, notebook_filepath: str, branch: str)` has been provided to generate this code.

Data Sources files are to be referenced from their online sources.  

### Python Package Setup Files
These files are used to build the Python package.
```txt
📦data-analytics-project-template
 ┣ 📂python-package
 ┃ ┣ 📂project_name
 ┃ ┃ ┣ 📜dataframe_labels.py
 ┃ ┃ ┣ 📜project_manager.py
 ┃ ┃ ┗ 📜__init__.py
 ┃ ┣ 📂tests
 ┃ ┃ ┗ 📜test_basic.py
 ┃ ┣ 📜license
 ┃ ┣ 📜pyproject.toml
 ┃ ┣ 📜readme.md
 ┃ ┣ 📜requirements.txt
 ┗ ┗ 📜setup.cfg
```
> Python Package Setup

### System Files

#### nbautoexport
The scripts folders in notebooks and samples and their contents are generated by [nbautoexport](install-nbautoexport.md) and [pipreqs](install-pipreqs.md) respectively. You should not need to modify these files. The `.nbautoexport` files are created by [nbautoexport](install-nbautoexport.md) and required for its functionality.

#### git
The [.gitattributes](./../.gitattributes) file is configured to use [pandoc](https://pandoc.org/) for comparing Microsoft Word .docx files.
The [.gitignore](./../.gitignore) file was created using [gitignore.io](https://www.toptal.com/developers/gitignore) and is used to ignore files that are not to be committed.

#### Windows folder icons
The files `desktop.ico` and `desktop.ini` are used to set the icon and name of the project folder on a Windows computer. The folder must be read-only for this setting to take effect.

```txt
📦data-analytics-project-template  
 ┣ 📂.vscode  
 ┣ 📂notebooks  
 ┃ ┣ 📂script  
 ┃ ┗ 📜.nbautoexport  
 ┣ 📜.gitattributes  
 ┣ 📜.gitignore  
 ┣ 📜desktop.ico  
 ┗ 📜desktop.ini  
```
> System Files


## Site Directory File List
```txt
📦data-analytics-project-template  
 ┣ 📂.vscode  
 ┃ ┗ 📜settings.json  
 ┣ 📂artifacts  
 ┃ ┣ 📜group-skills.xlsx  
 ┃ ┗ 📜readme.md  
 ┣ 📂assets  
 ┃ ┣ 📜2021-12Dec-11-population-estimates-1950-2021-pea01.csv  
 ┃ ┗ 📜readme.md  
 ┣ 📂docs  
 ┃ ┣ 📂images
 ┃ ┃ ┣ 📜correlation-matrix-heatmap-pyramid.png
 ┃ ┃ ┗ 📜gantt-chart.jfif
 ┃ ┣ 📜assessment-criteria.md
 ┃ ┣ 📜build-python-package.md
 ┃ ┣ 📜build-requirements.md
 ┃ ┣ 📜code-style-guide.md
 ┃ ┣ 📜gantt-chart.md
 ┃ ┣ 📜install-nbautoexport.md
 ┃ ┣ 📜install-python-package.md
 ┃ ┣ 📜jupyter-notebook-layout-guide.md
 ┃ ┣ 📜knowledge-skills-abilities.md
 ┃ ┣ 📜notebook-managers.md
 ┃ ┣ 📜project-structure-guide.md
 ┃ ┣ 📜readme.md
 ┃ ┗ 📜template-todo.md
 ┣ 📂junk-dna  
 ┃ ┗ 📜readme.md  
 ┣ 📂notebooks  
 ┃ ┣ 📂script  
 ┃ ┃ ┣ 📜notebook-1-01-example-bad-code-population.py  
 ┃ ┃ ┣ 📜notebook-2-01-example-better-code-population-eda.py  
 ┃ ┃ ┣ 📜notebook-2-02-example-better-code-population-dv.py  
 ┃ ┃ ┗ 📜requirements.txt  
 ┃ ┣ 📜.nbautoexport  
 ┃ ┣ 📜notebook-1-01-example-bad-code-population.ipynb  
 ┃ ┣ 📜notebook-2-01-example-better-code-population-eda.ipynb  
 ┃ ┣ 📜notebook-2-02-example-better-code-population-dv.ipynb  
 ┃ ┗ 📜readme.md  
 ┣ 📂python-package  
 ┃ ┣ 📂project_name  
 ┃ ┃ ┣ 📜dataframe_labels.py  
 ┃ ┃ ┣ 📜project_manager.py  
 ┃ ┃ ┗ 📜__init__.py  
 ┃ ┣ 📂tests  
 ┃ ┃ ┗ 📜test_basic.py  
 ┃ ┣ 📜license  
 ┃ ┣ 📜pyproject.toml  
 ┃ ┣ 📜readme.md  
 ┃ ┣ 📜requirements.txt  
 ┃ ┗ 📜setup.cfg  
 ┣ 📂references  
 ┃ ┣ 📜2021-11nov-03-data-scientists-your-variable-names-are-awful.md  
 ┃ ┣ 📜2021-11nov-03-pep-515-underscores-in-numeric-literals.md  
 ┃ ┣ 📜2021-11nov-03-pep-8-style-guide-for-python-code.md  
 ┃ ┣ 📜2021-11nov-04-python-3-f-strings.md  
 ┃ ┣ 📜2021-11nov-04-to-camel-case-or-under-score.pdf  
 ┃ ┗ 📜readme.md  
 ┣ 📜.gitattributes  
 ┣ 📜.gitignore  
 ┣ 📜desktop.ico  
 ┣ 📜desktop.ini  
 ┣ 📜license  
 ┗ 📜readme.md  
```

&nbsp;

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.