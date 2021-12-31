# [Data Analytics Project Template](./../../../)

## Code Style Guide

> "This appears to be a region of space that doesn't have many rules. But I believe we can learn something from the events that have unfolded. In a part of space where there are few rules, it's more important than ever that we hold fast to our own." – Captain Kathryn Janeway, 2372

The Python Style guide states that "The naming conventions of Python's library are a bit of a mess, so we'll never get this completely consistent". camelCase was considered for easier readability [(Dave Binkley, 2009)](./../references/readme.md#DaveBinkley2009) however because this project is a data science portfolio the Python code shall be written to confirm to the Python Style guide as much as possible [(Guido van Rossum, 2013)](./../references/readme.md#GuidoVanRossum2013). [(Koehrsen, 2019)](./../references/readme.md#Koehrsen2019) discusses the problem of awful variable names in data science. These observations have been used in crafting the style guide.

### Naming conventions

Use meaningful variable names. The variable name must describe the information represented by the variable. A variable name should tell you specifically in words what the variable stands for [(Koehrsen, 2019)](./../references/readme.md#Koehrsen2019) e.g., `dataframe` rather than `df`, and `index` rather than `i`.

Avoid these awful variable names:  

- Single letter names
- Unhelpful/confusing/vague variable names
- Obscure Acronyms
- Abbreviations
- Meaningless names

When creating an object of class, the name should generally include the class name.

**Examples:**
```python
dataframe = pandas.parse_csv(filename)
customer_dataframe = pandas.parse_csv(customer_filename)
stock_dataframe = pandas.parse_csv(stock_filename)
```

**Incorrect Examples**

| Wrong | Reason |
|-|-|
| `df = pandas.parse_csv(filename)` | Abbreviation, `df` is a not a descriptive name |
| `customer_df = pandas.parse_csv(customer_filename)` | Ambiguous name. What is a `customer_df`, what fields and methods would you expect? |
| `df_stock = pandas.parse_csv(stock_filename)` | `df_stock` is Hungarian notation |  

#### Project code abbreviations

abbreviations|term
-|-
iqr | Inter-quartile range
max | maximum
min | minimum
q1, q2, q3 | In context quartile

### Avoid magic constant numbers

The term magic number or magic constant refers to the anti-pattern of using numbers directly in source code. These numbers lack explanation, and their purpose requires fully understanding their context. If the constant is changed, multiple edits are required to the source code.

Contrast Example:

```python
def calculate_limits(dataframe: DataFrame, column_label: str) -> tuple:
    q1 = *dataframe[column_label].quantile(0.25)
    q3 = *dataframe[column_label].quantile(0.75)
    iqr = q3 - q1
    lowerLimit = q1 – 1.5 * iqr
    upperLimit = q3 + 1.5 * iqr
return lowerLimit, upperLimit
```

> What is 1.5? Why 1.5 and not 1 or 2?

```python
def calculate_limits(dataframe: DataFrame, column_label: str) -> tuple:
    Q1 = 0.25
    Q3 = 0.75
    TUKEY_IQR_SCALE = 1.5

    q1 = *dataframe[column_label].quantile(Q1)
    q3 = *dataframe[column_label].quantile(Q3)
    iqr = q3 - q1
    lowerLimit = q1 – TUKEY_IQR_SCALE * iqr
    upperLimit = q3 + TUKEY_IQR_SCALE * iqr
    return lowerLimit, upperLimit
```

> A quick search of TUKEY_IQR_SCALE details John Tukey's 1.5 IQR rule and the explanation its value.

### Python variable naming

#### Local variables

Local variables are to be defined in lower snake_case. camelCase isn't recommended but as they are local it won't be an issue.

**Examples:**
```python
filename = 'births-by-county-1985-2020.csv'
velocity_avg = 78
column_label = 'First name'
dataframe = pd.read_csv(filename)
iqr = 2.43344
roi_counties = ['Clare', 'Limerick']
```

**Incorrect Examples**

| Wrong | Reason |
|-|-|
| `fileName = 'data.csv'` | camelCase|
| `Velocity_AVG = 78` | A mix of PascalCase and snake_case|
| `columnLabel = 'First name'` | camelCase|
| `Dataframe = pd.read_csv(filename)` | PascalCase|
| `IQR` | Uppercase, looks like a constant|
| `ROI_counties = ['clare', 'limerick']` | Uppercase abbreviation and of mix snake case |

#### Constant Variables

Constants are to be name in UPPER_SNAKE_CASE

```python
Q1 = 0.25
Q3 = 0.75
PIE = 3.14
IRISH_COUNTY_COUNT = 26
```

#### Booleans

Boolean variable generally begin with is or has

**Examples:**  

```python
is_remote = False
has_limits = True
```

**Incorrect Examples**  

| Wrong | Reason |
|-|-|
| `remote = False` | Without the value of the variable is unclear what remote is |
| `limits = True` | Without the value of the variable is unclear what the limits are.  <br/>By Beginning a name with a plural word, it suggests it is an array/number |

### Methods

Methods are to be defined with snake_case as they are part of the API. Method declarations should include the type of the parameters and the return type.

**Example:**  

```python
def get_github_url(repository_url: str, filename: str) -> str:
    return f'{repository_url}/{filename}'


def print_dataframe(dataframe: DataFrame) -> None:
    display(dataframe.head())
```

**Incorrect Example:**  

```python
def get_github_url(repository_url, filename):
    return f'{repository_url}/{filename}'


def print_dataframe(dataframe):
    display(dataframe.head())
```

### Imports

Python Imports tend to be given short acronyms e.g., import pandas as pd. Given the existing convention and many examples available using these conventions imports alias can be used at the programmer's discretion. No unused imports are to be present in the file.

### Comments

> "A comment is a failure to express yourself in code. If you fail, then write a comment; but try not to fail." And "If the "why" is not obvious, and you cannot make it obvious with code, use a comment." - [Uncle Bob Martin, 2017](https://twitter.com/unclebobmartin/status/870311898545258497).

Use inline comments sparingly. [(Guido van Rossum, 2013)](./../references/readme.md#GuidoVanRossum2013)

Inline comments are unnecessary and in fact distracting if they state the obvious. Don't do this:

```python
x = x + 1     # Increment x
```
**But sometimes, this is useful:** 

```python
x = x + 1    # Compensate for border
```

### Documentation

Methods should be documented explaining what they do and what the parameters are.
```python
def create_jupyter_notebook_header(github_username: str, repository: str, notebook_filepath: str, branch: str = 'master') -> str:
    """
    create an edit-online header for Jupyter Notebook
    :param github_username: GitHub username
    :param repository: repository name
    :param notebook_filepath: notebook file path
    :param branch: branch name
    :return: HTML header
    """
```

### Classes

Class names are to be defined with Pascal Case.

### Spacing, Indentation

In python code, the IDEs shall manage spacing and indentation. Spaces shall be used, not tabs. In Jupyter no indentation policy shall be enforced.

### Literals 

#### Sting Literals

Use f-Strings. Clearer syntax and they are fasted than % and str.format [(Jablonski, 2018)](./../references/readme.md#Jablonski2018) 

```python
repository_url = "https://www.github.com/markcrowe-com"
filename = "docs/population-planning-data-analytics.docx"
```

Avoid broken up strings

```python
url = repository_url + "/" + filename
```

Use f-Strings.

```python
url = f"{repository_url}/{filename}"
```
#### Numeric Literals

For large numbers use numeric literals. [(Georg Brandl, 2016)](./../references/readme.md#GeorgBrandl2016)

```python
# Grouping decimal numbers by thousands
amount = 10_000_000.0

# Grouping hexadecimal addresses by words
address = 0xCAFE_F00D

# Grouping bits into nibbles in a binary literal
flags = 0b_0011_1111_0100_1110
```

If specifying a numeric type (binary, octal, hexadecimal) use lowercase
`b`, `o`, `x` to avoid confusion with numbers like `0` and `8`  

[References](./../references/readme.md)

---
**Template footnote**  
This project started from the template <https://github.com/markcrowe-com/data-analytics-project-template>. Permission is granted to reproduce for personal and educational use only. Commercial copying, hiring, lending is prohibited. In all cases this notice must remain intact. Author [Mark Crowe](https://github.com/markcrowe-com/) Copyright &copy; 2021, All rights reserved.