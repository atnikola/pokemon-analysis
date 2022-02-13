# Pokemon Analysis
Andreas Nikolaidis
February 2022 - Jupyter Notebook (Link)

- [Introduction](docs/README.md/#Introduction)
- [Exploratory Analysis](docs/README.md)
- [Correlation](docs/README.md)
- [Principal Component Analysis (PCA)](docs/README.md)
- [Cross Validation](docs/README.md)
- [Multiple Regression Analysis](docs/README.md)
- [Conclusion](docs/README.md)

## Introduction
In this project, I use Python to analayze stats on all Pokemon in Generations 1 - 8, and calculate some interesting statistics based on a number of factors. 

We can use this data to answer questions such as:
- **Does a Pokemon's Type determine it's stats like: HP, Attack, Defense, etc.?**
- **What is the most important stat for predicting other stats? i.e. which stats have a high correlation?**

In the following sections, I will walk through my process of extracting and analyzing the information using in ```pandas DataFrames```, creating some visualizations and perform modeling using ```scikit-learn```.

## Exploratory Analysis
Start by importing all the necessary packages into Python:
```python
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px
import plotly.graph_objects as go

sns.set_style('whitegrid')
%matplotlib inline

# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
```

Next we read in the excel file containing the data & check the head of the data:
```python
df = pd.read_excel("pokemon.xlsx")
df.head()
```
Create a separate dataframe including just the necessary stats:
```python
df_stats = df[["Name","HP","Attack","Defense","SP_Attack","SP_Defense","Speed"]]
```
Although each stat is important in it's own right, the total value of all stats is what determines the category of a pokemon, therefore let's concatenate a column into the ```df``` that sums up the total values:
```python
df['total'] = df.HP + df.Attack + df.Defense + df.SP_Attack + df.SP_Defense + df.Speed
df.columns
```



