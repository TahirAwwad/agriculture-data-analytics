# [Agriculture Data Analytics](./../../../)

# Build and publish a Python package

*All these commands must be run from the `python-package` folder path in the project:*

## Build/rebuild the Python package
To build the Python package, run the following command:
```bash
python -m build
```

## Publish the Python package
To publish the package to PyPI, run the following command:
```bash
python -m twine upload dist/*
```
For username enter `__token__` and then your password.  

The package is then available at <https://pypi.org/project/PACKAGE_URL/>

## Installation of the Python package

### Remote installation

To install the package from [pypi.org](https://pypi.org), run the following command:

```bash
pip install PACKAGE_NAME
```

### Local installation
To install the package from local sources, run the following command:

```bash
pip install .\dist\PACKAGE_NAME-0.0.1-py3-none-any.whl
```

To force a reinstall of the package from local sources, run the following command:

```bash
pip install .\dist\PACKAGE_NAME-0.0.1-py3-none-any.whl --force-reinstall
```

Conda:  
```bash
conda install .\dist\PACKAGE_NAME-0.0.1-py3-none-any.whl --channel conda-forge
```

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.