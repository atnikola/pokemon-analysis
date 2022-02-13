# Pokémon Analysis
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
```
Create a separate dataframe including just the necessary stats:
```python
df_stats = df[["Name","HP","Attack","Defense","SP_Attack","SP_Defense","Speed"]]
```
Although each stat is important in it's own right, the total value of all stats is what determines the category of a pokemon, therefore let's concatenate a column into the ```df``` that sums up the total values:
```python
df['total'] = df.HP + df.Attack + df.Defense + df.SP_Attack + df.SP_Defense + df.Speed
```

Now let's view the range of total stats by each generation:
```python
#palette: https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=color
plt.figure(figsize=(13,10), dpi=80)
sns.violinplot(x='Gen', y='total', data=df, scale='width', inner='quartile', palette='Set2') 
plt.title('Violin Plot of Total Stats by Generation', fontsize=22)
plt.show()
```
![2df65225-732a-4581-af16-46cbaf14b931](https://user-images.githubusercontent.com/38530617/153741822-44e70858-0ce7-436c-b649-5c172f4ce08f.png)

In the above violinplot we can see that each generation has quite a different range of total stats with Gens IV, VII, & VIII having the longest range, while Gen V had a relatively tight range of stats. All Generations from IV onwards had higher medians than the first 3 generations. 

Looking at individual stats, **Speed** is one of (if not THE) most important stat in competitive play, so let's examine which generations had the best overall speed stats.

```python
plt.figure(figsize=(13,10), dpi=80)
sns.violinplot(x='Gen', y='Speed', data=df, scale='width', inner='quartile', palette='Set2')

plt.title('Violin Plot of Total Stats by Generation', fontsize=22)
plt.show()
```
![speed](https://user-images.githubusercontent.com/38530617/153742342-dfcc2c47-541f-44f2-8998-a04a83ebe10d.png)

Here we can clearly see Generation VIII has some of the fastest pokemon ever seen in games. Let's create a function to return the top 10 fastest pokemon in Gen VIII and their respective speed stat values:

```python
def top_n(df, category, n):
    return (df.loc[df['Gen'] == 'VIII'].sort_values(category, ascending=False)[['Name','Gen',category]].head(n))
```
```python
print('Top 10 Pokemon Speed')
top_n(df, 'Speed', 10)
```
<img width="238" alt="speed_gen8" src="https://user-images.githubusercontent.com/38530617/153742615-a6d3e5a6-12d6-45de-bef5-4377deef61e7.png">




