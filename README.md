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

Those are definitely some fast pokemon!

Let's now see if we can get any indication of whether a particular pokemon's type has an advantage over others in total stats.

```python
types_color_dict = {
    'grass':'#8ED752', 'fire':'#F95643', 'water':'#53AFFE', 'bug':"#C3D221", 'normal':"#BBBDAF", \
    'poison': "#AD5CA2", 'electric':"#F8E64E", 'ground':"#F0CA42", 'fairy':"#F9AEFE", \
    'fighting':"#A35449", 'psychic':"#FB61B4", 'rock':"#CDBD72", 'ghost':"#7673DA", \
    'ice':"#66EBFF", 'dragon':"#8B76FF", 'dark':"#1A1A1A", 'steel':"#C3C1D7", 'flying':"#75A4F9" }

plt.figure(figsize=(15,12), dpi=80)
sns.violinplot(x='Primary', y='total', data=df, scale='width', inner='quartile', palette=types_color_dict)

plt.title('Violin Plot of Total Stats by Type', fontsize=20)
plt.show()
```
![total_type_stats](https://user-images.githubusercontent.com/38530617/153742776-8ff03580-972e-4b37-a9cb-daed42e09163.png)

The **dragon type** definitely has quite a high upper interquartile range compared to other types. Meanwhile water & fairy types seem to have quite a large variance in total stats. 

Let's see what the most common type of pokemon is:
```python
types_color_dict = {
    'grass':'#8ED752', 'fire':'#F95643', 'water':'#53AFFE', 'bug':"#C3D221", 'normal':"#BBBDAF", \
    'poison': "#AD5CA2", 'electric':"#F8E64E", 'ground':"#F0CA42", 'fairy':"#F9AEFE", \
    'fighting':"#A35449", 'psychic':"#FB61B4", 'rock':"#CDBD72", 'ghost':"#7673DA", \
    'ice':"#66EBFF", 'dragon':"#8B76FF", 'dark':"#1A1A1A", 'steel':"#C3C1D7", 'flying':"#75A4F9" }


Type1 = pd.value_counts(df['Primary'])
sns.set()
dims = (11.7,8.27) #A4 dimensions
fig, ax=plt.subplots(figsize=dims)
BarT = sns.barplot(x=Type1.index, y=Type1, data=df, palette=types_color_dict, ax=ax)
BarT.set_xticklabels(BarT.get_xticklabels(), rotation= 90, fontsize=12)
BarT.set(ylabel = 'Freq')
BarT.set_title('Distribution of Primary Pokemon Types')
FigBar = BarT.get_figure()
```
![type_distribution](https://user-images.githubusercontent.com/38530617/153743003-888b9e1b-4353-4cb8-b07f-bfd93e513c29.png)

We can see that the water and normal type pokemon are the most frequently appearing 'primary' types in the game.

Let's see how many pokemon are mono types vs dual-types so we can get a better sense of whether primary is sufficient.

```python
labels = ['Mono type pokemon', 'Dual type pokemon']
sizes = [monotype, dualtype]
colors = ['lightskyblue', 'lightcoral']

patches, texts, _ = plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0,0.1))
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.title('Dual-Type Ratio', fontsize=12)
plt.tight_layout()
plt.show()
```
![mono_dual](https://user-images.githubusercontent.com/38530617/153743106-853f1990-c3d3-4cfb-8db7-dd119929d39c.png)

**Looks like there's actually more dual types than mono-types!**

Aside from types, there are also 5 categories of pokemon: Regular, Pseudo-Legendary, Sub-Legendary, Legendary and Mythical. (There are of course also pre-evolutions, final evolutions, mega-evolutions etc.. but for the purposes of this analysis we will just bundle those together under 'regular' along with Pseudo-Legendary which are regular pokemon that have generally higher stats of 600 total. 
As for Sub Legendaries, Legendaries and Mythical - these pokemon typically exhibit **2 types of traits**: 
1. Rarity: There is usually only 1 of those pokemon available in every game (some may not even be obtainable in certain games)
2. Stats: These pokemon generally have much higher stats than the average 'regular' pokemon.

Let's create a diverging bar to determine the rate at which these pokemon appear in each generation:
```python
#Sub-Legendary, Legendary or Mythical:
df.loc[df["is_sllm"]==False,"sllmid"] = 0
df.loc[df["is_sllm"]==True,"sllmid"] = 1

# calculate proportion of SL, L, M #
sllm_ratio = df.groupby("Gen").mean()["sllmid"]
sllm_ratio.round(4)*100
```



```python
sns.set_style('darkgrid')
df_plot = pd.DataFrame(columns={"Gen","Rate","colors"})
x = sllm_ratio.values
df_plot["Gen"] = sllm_ratio.index
df_plot['Rate'] = (x - x.mean())/x.std()
df_plot['colors'] = ['red' if x < 0 else 'green' for x in df_plot['Rate']]
df_plot.sort_values('Rate', inplace=True)
df_plot.reset_index(inplace=True)

plt.figure(figsize=(14, 10))
plt.hlines(
    y=df_plot.index, xmin=0, xmax=df_plot.Rate,
    color=df_plot.colors,
    alpha=.4,
    linewidth=5)

plt.gca().set(xlabel='Rate', ylabel='Gen')
plt.yticks(df_plot.index, df_plot.Gen, fontsize=12)
plt.title('Diverging Bars of SubL, Legendary & Mythical Rate', fontdict={'size':20})
plt.show()
```

![sub, legend,myth](https://user-images.githubusercontent.com/38530617/153743612-15eb0d03-0606-4f18-9bcf-9215b963f79e.png)

Seems like Gen 7's **Alola** region has a huge volume of these 'legendaries & mythical' pokemon, which after digging further into it makes perfect sense given the introduction of a plethora of legendaries called **'ultra beasts'** which were only ever introduced in that generation.














