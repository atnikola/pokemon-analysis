# Pokémon Analysis
**Andreas Nikolaidis** 

_February 2022_ (_Edited: September 2023_)

- [Introduction](#introduction)
- [Exploratory Analysis](#exploratory_analysis)
- [Correlations & Descriptive Statistics](#descriptive)
- [Principal Component Analysis (PCA)](#pca)
- [Cross Validation & Regression Analysis](#cv-ra)
- [Conclusion](#conclusion)

## [Introduction](#introduction)
In this project, I will aim to analyze the stats of all Pokemon in Generations 1 - 9, and calculate some statistics based on a number of factors.
In the following sections, I will walk through my process of extracting and analyzing the information using ```pandas```, creating some visualizations and modeling using ```scikit-learn```.

## [Exploratory Analysis](#exploratory_analysis)
Start by importing all the necessary packages into Python:
```python
import numpy as np
import pandas as pd

# Viz
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

sns.set_style('whitegrid')
%matplotlib inline

# Import for Linear Regression
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
```

Read Data File:
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
```
```python
df.head(3).style.bar(subset=['Total', 'HP', 'Attack', 'Defense', 'SP_Attack', 'SP_Defense', 'Speed'])
```
```python
#Create a dataframe of just the main stats excluding other 'non important' variables
df_stats = df[["Name","HP","Attack","Defense","SP_Attack","SP_Defense","Speed"]]
```

**Visuals**

Now let's view the range of total stats by each generation:
```python
plt.figure(figsize=(13,10), dpi=80)
sns.violinplot(df, x='Gen', y='Total', scale='width', inner='quartile', palette='pastel') 
#palette: https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=color

plt.title('Violin Plot of Total Stats by Generation', fontsize=22)
plt.show()
```
![3449927f-f35f-4049-9687-e79ec7ec7733](https://github.com/atnikola/pokemon-analysis/assets/38530617/d54ed598-9c52-4f91-8c48-a446ee757078)

In the above violinplot we can see that each generation has quite a different range of total stats with Gens IV, VII, & VIII having the longest range, while Gen V had a relatively tight range of stats. All Generations from IV onwards had higher medians than the first 3 generations.

Looking at individual stats, Speed is one of (if not THE) most important stat in competitive play, so let's examine which generations had the best overall speed stats.

```python
plt.figure(figsize=(13,10), dpi=80)
sns.violinplot(df, x='Gen', y='Speed', scale='width', inner='quartile', palette='pastel')

plt.title('Violin Plot of Speed Stats by Generation', fontsize=22)
plt.show()
```
![c9a2dfd0-832a-4f7e-bde1-9e62783d8073](https://github.com/atnikola/pokemon-analysis/assets/38530617/99858c36-8213-469f-8132-083ef9124c64)

Here we can clearly see Generation VIII has some of the fastest pokemon ever seen in games. Let's create a function to return the top 10 fastest pokemon in Gen VIII and their respective speed stat values:

```python
def top_n(df, category, n):
    return (df.loc[df['Gen'] == 'VIII'].sort_values(category, ascending=False)[['Name','Gen',category]].head(n))
```
```python
print('Top 10 Pokemon Speed')
top_n(df, 'Speed', 10)
```
<img width="240" alt="Screenshot 2023-09-20 at 23 23 24" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/d24e9674-2156-4501-afb7-7374cfb5666b">

Those are definitely some fast pokemon!

Let's now see if we can get any indication of whether a particular pokemon's type has an advantage over others in total stats.

```python
#custom colors based on color of types from games
types_color_dict = {
    'grass':'#8ED752', 'fire':'#F95643', 'water':'#53AFFE', 'bug':"#C3D221", 'normal':"#BBBDAF", \
    'poison': "#AD5CA2", 'electric':"#F8E64E", 'ground':"#F0CA42", 'fairy':"#F9AEFE", \
    'fighting':"#A35449", 'psychic':"#FB61B4", 'rock':"#CDBD72", 'ghost':"#7673DA", \
    'ice':"#66EBFF", 'dragon':"#8B76FF", 'dark':"#1A1A1A", 'steel':"#C3C1D7", 'flying':"#75A4F9" }


plt.figure(figsize=(30,12), dpi=80)
sns.violinplot(df, x='Primary', y='Total', scale='width', inner='quartile', palette=types_color_dict)

plt.title('Violin Plot of Total Stats by Type', fontsize=15)
plt.show()
```
![e03d6822-c752-41bc-9177-1b1227d64807](https://github.com/atnikola/pokemon-analysis/assets/38530617/de20b320-b81d-484d-8dc6-5d1d0a1e44f3)

The **dragon type** definitely has quite a high upper interquartile range compared to other types, which makes sense as many legendary are dragon type. Meanwhile water & fairy types seem to have quite a large variance in total stats. 

Let's see what the most common type of pokemon is:
```python
Type1 = pd.value_counts(df['Primary'])
sns.set()
dims = (25,8)
fig, ax=plt.subplots(figsize=dims)
BarT = sns.barplot(df, x=Type1.index, y=Type1, palette=types_color_dict, ax=ax)
BarT.set_xticklabels(BarT.get_xticklabels(), rotation= 90, fontsize=12)
BarT.set(ylabel = 'Frequency')
BarT.set_title('Distribution of Primary Pokemon Types')

##Annotate values
for bar in BarT.patches:
    BarT.annotate(format(bar.get_height(), '.0f'),
                   (bar.get_x() + bar.get_width() / 2,
                    bar.get_height()), ha='center', va='center',
                   size=10, xytext=(0, 8),
                   textcoords='offset points')

FigBar = BarT.get_figure()
```

![9d8c1816-81cf-4677-9e23-fedb6cc29256](https://github.com/atnikola/pokemon-analysis/assets/38530617/6aadfa44-f230-4fd7-9614-c7f4d4199e6f)

We can see that the water and normal type pokemon are the most frequently appearing 'primary' types in the game. Interesting to see Flying types as lowest however it makes sense when we only look at primary types as majority of pokemon that are dual types with Flying usually have flying as their 'secondary' type. Meaning, a Pokemon is normally not "Flying/Normal", it's most commonly: "Normal/Flying" for example. 

Let's see how many pokemon are mono types vs dual-types so we can get a better sense of whether primary is sufficient.

A simple method would be to do a count over but lets create a chart:

```python
labels = ['Mono type pokemon', 'Dual type pokemon']
sizes = [monotype, dualtype]
colors = ['lightskyblue', 'lightcoral']

patches, texts, _ = plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0,0.1))
plt.legend(patches, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.axis('equal')
plt.title('Dual-Type Ratio', fontsize=12)
plt.tight_layout()
plt.show()
```
![0a54a33b-da39-47ef-8344-f6dca35cf9de](https://github.com/atnikola/pokemon-analysis/assets/38530617/8a24c134-3e20-473c-90f5-486c64ef75de)

**Looks like there's actually more dual types than mono-types**

Aside from types, there are also 5 categories of pokemon: Regular, Pseudo-Legendary, Sub-Legendary, Legendary and Mythical. (There are of course also pre-evolutions, final evolutions, mega-evolutions etc.. but for the purposes of this analysis we will just bundle those together under 'regular' along with Pseudo-Legendary which are regular pokemon that have generally higher stats of 600 total. 
As for Sub Legendaries, Legendaries and Mythical - these pokemon typically exhibit **2 types of traits**: 
1. Rarity: There is usually only 1 of those pokemon available in every game (some may not even be obtainable in certain games)
2. Stats: These pokemon generally have much higher stats than the average 'regular' pokemon.

Let's create a diverging bar to determine the rate at which legendary pokemon appear in each generation:
```python
#Sub-Legendary, Legendary or Mythical:
df.loc[df["is_sllm"]==False,"sllmid"] = 0
df.loc[df["is_sllm"]==True,"sllmid"] = 1

# calculate proportion of SL, L, M #
sllm_ratio = df.groupby("Gen").mean()["sllmid"]
sllm_ratio.round(4)*100
```
<img width="107" alt="Screenshot 2023-09-20 at 23 36 03" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/fb6bf1e4-abd1-4417-a788-04284f98505a">

```python
sns.set_style('darkgrid')
df_plot = pd.DataFrame(columns=["Gen", "Rate", "colors"])  # Use square brackets [] here
x = sllm_ratio.values
df_plot["Gen"] = sllm_ratio.index
df_plot['Rate'] = (x - x.mean()) / x.std()
df_plot['colors'] = ['red' if x < 0 else 'green' for x in df_plot['Rate']]
df_plot.sort_values('Rate', inplace=True)
df_plot.reset_index(inplace=True)

plt.figure(figsize=(10, 10))
plt.hlines(
    y=df_plot.index, xmin=0, xmax=df_plot.Rate,
    color=df_plot.colors,
    alpha=.4,
    linewidth=5)

plt.gca().set(xlabel='Rate', ylabel='Gen')
plt.yticks(df_plot.index, df_plot.Gen, fontsize=12)
plt.title('Diverging Bars Rate', fontdict={'size': 20})
plt.show()
```
![4292a7a5-2046-49f4-a8c0-336bd274f3ca](https://github.com/atnikola/pokemon-analysis/assets/38530617/26a83d37-6d44-4215-a8aa-bbab928bf97b)

Seems like Gen 7's **Alola** region has a huge volume of these 'legendaries & mythical' pokemon, which after digging further into it makes perfect sense given the introduction of a plethora of legendaries called **ultra beasts** which were only ever introduced in that generation.

## [Correlations & Descriptive Statistics](#descriptive)
Let's move to explore some correlations between stats.

```python
from pandas import plotting
plotting.scatter_matrix(df_stats, figsize=(10, 10)) 
plt.show()
```
![1cbcd222-5f80-4669-9e57-73ced0d6c60c](https://github.com/atnikola/pokemon-analysis/assets/38530617/2338fb4e-4a9a-4b15-a90a-f9e09547c83c)

```python
corrcoef = np.corrcoef(df_stats.iloc[:, 1:7].T.values.tolist())
plt.imshow(corrcoef, interpolation='nearest', cmap=plt.cm.magma)
plt.colorbar(label='correlation coefficient')
tick_marks = np.arange(len(corrcoef))
plt.xticks(tick_marks, df_stats.iloc[:, 1:7].columns, rotation=90)
plt.yticks(tick_marks, df_stats.iloc[:, 1:7].columns)
plt.tight_layout() #clean
```
![d7cb19bc-162e-485e-b2ca-c495411949e9](https://github.com/atnikola/pokemon-analysis/assets/38530617/0f9c0800-d65c-4715-99c9-39da7355297f)

```python
###----Correlations
Base_stats = ['Primary','Secondary','Height','Weight','HP','Attack','Defense',
              'SP_Attack','SP_Defense','Speed','is_sllm']

df_BS = df[Base_stats]

plt.figure(figsize=(14,12))

heatmap = sns.heatmap(df_BS.corr(), vmin=-1,vmax=1, annot=True, cmap='Blues')

heatmap.set_title('Correlation Base Stats Heatmap', fontdict={'fontsize':15}, pad=12)
plt.show()

```
![f59aa5df-bdfa-4107-94a0-2699f1f48c53](https://github.com/atnikola/pokemon-analysis/assets/38530617/2057e5cd-7e69-49b7-b64a-a5abef6a7731)

Some other charts showing stat correlations:

![e946888c-5d0b-476a-872e-ebfe582b4957](https://github.com/atnikola/pokemon-analysis/assets/38530617/89745bb6-661f-49f6-836d-a7f4d06be3fa)
![c96caeb7-4b7a-4906-8ae8-eceaeffd28a8](https://github.com/atnikola/pokemon-analysis/assets/38530617/36f46590-12bb-4c2b-a5d5-d502d4556cef)
![be606f8a-ef76-494a-8239-951e78241d11](https://github.com/atnikola/pokemon-analysis/assets/38530617/00f0533e-f5ac-4dc9-b37e-594d1f85e73a)
![863627ea-37f4-490c-8e9b-f1f60dbcbcfe](https://github.com/atnikola/pokemon-analysis/assets/38530617/95d682bd-acc1-4bbe-9d87-bf2bab006951)
![40db912c-61c7-4d9b-9584-27c845e59fef](https://github.com/atnikola/pokemon-analysis/assets/38530617/222a7ce5-dd0e-4268-9b33-d4666d0881ae)

Let's take a look at diverging bars based on Attack to see what pokemon type stands out in that specific stat:

```python
attack_byType = df.groupby("Primary").mean()["Attack"]

df_plot2 = pd.DataFrame(columns=["Type","Attack","colors"]) #square brackets
x = attack_byType.values
df_plot2["Type"] = attack_byType.index
df_plot2['Attack'] = (x - x.mean())/x.std()
df_plot2['colors'] = ['red' if x < 0 else 'green' for x in df_plot2['Attack']]
df_plot2.sort_values('Attack', inplace=True)

plt.figure(figsize=(14,14), dpi=80)
plt.hlines(y=df_plot2.Type, xmin=0, xmax=df_plot2.Attack)
for x, y, tex in zip(df_plot2.Attack, df_plot2.Type, df_plot2.Attack):
    t = plt.text(
        x, y, round(tex, 2),
        horizontalalignment='right' if x < 0 else 'left',
        verticalalignment='center',
        fontdict={'color':'red' if x < 0 else 'green', 'size':15})

plt.yticks(df_plot2.Type, df_plot2.Type, fontsize=12)
plt.title('Diverging Text Bars of Attack by Type', fontdict={'size':20})
plt.xlim(-3, 3)
plt.show()
```
![accd1eab-c264-427d-b171-e4d9a9a13fc1](https://github.com/atnikola/pokemon-analysis/assets/38530617/22285150-b394-4d0e-8532-86c0f81a8cf6)

**Seems like FIGHTING type is the most common**

Let's take a look at distribution of stats:
```python
df.describe()
```
<img width="1135" alt="Screenshot 2023-09-20 at 23 49 46" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/423caa07-d0ec-445c-8fe0-90c189679924">

![81c966c7-e2c7-4e8e-b355-4bc196d4dfa1](https://github.com/atnikola/pokemon-analysis/assets/38530617/fbed4f8f-e6e9-418d-b5c3-d14a40f5b5e1)

```python
labels = ["Defense", "Attack"]
dims = (11.7, 8.27) #a4
fig, ax = plt.subplots(figsize=dims)
Defhist = sns.distplot(df['Defense'],color='g', hist=True, ax=ax)
Atthist = sns.distplot(df['Attack'],color='r', hist=True, ax=ax)
Atthist.set(title='Distribution of Defense & Attack')
plt.legend(labels, loc="best")
FigHist = Atthist.get_figure()
```
![cb95c0fc-7cf7-4506-87fb-e6d10a641335](https://github.com/atnikola/pokemon-analysis/assets/38530617/addb9392-d30b-43e1-bb4b-f2a2df7e2a61)

By calling on the summary statistics, we can see that the assumption about the variance and skewness of both plots was correct. The ‘std’ metric (standard deviation) of 'Sp.Atk is larger than that of the Sp.Def. Skewness is determined by the positions of the median (50%) and the mean. Since in all instances (Attack, Defense, Sp.Attack and Sp.Defense) the mean is greater than the median, it is emphasised that the distribution is right-skewed (positively skewed).

# [Principal Component Analysis (PCA)](#pca)
Let's take a look at PCA and plot Pokemon in a two-dimensional plane using the first and second principal components.
PCA is a type of multivariate analysis method that is often used as a dimensionality reduction method and sometimes regarded as a type of **unsupervised ML**, revealing the structure of the data itself. 

In this data, the characteristics of all Pokemon total stats are represented by **6 types** of **"observed variables"** (x1, x2, x3, x4, x5, x6). 
(As explained earlier - HP, Attack, Defense, SP Attack, SP Defense & Speed) - these 6 variables are used as explanatory variables. 
On the other hand, the synthetic variable synthesized by PCA is called "principal component score" and is given by a linear combination as shown in the following equation: 

yPC1 = a1,1 x1 + a1,2 x2 + a1,3 x3 + a1,4 x4 + a1,5 x5 + a1,6 x6

In principal component analysis, the larger the ```eigenvalue``` (= variance of the principal component score), **the more important the principal component score is**.

```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df.iloc[:, 7:13])
feature = pca.transform(df.iloc[:, 7:13])
plt.figure(figsize=(8,8))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
#plt.show()
```
![1bec3a8d-e2e0-43e6-aa27-a26a6633d26b](https://github.com/atnikola/pokemon-analysis/assets/38530617/b8f3d97b-eaed-43e9-9fc3-77e85f69c68f)


```python
import matplotlib.ticker as ticker

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("# PC")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
#plt.show()
```
![21a2ac5a-825c-4a24-ad2e-314905c00dcf](https://github.com/atnikola/pokemon-analysis/assets/38530617/89470f4d-9f07-4b80-834c-b58926b7d225)

    
Let's see if we can determine what makes a pokemon **'LEGENDARY'**

```python
pca = PCA()
pca.fit(df.iloc[:, 7:13])
feature = pca.transform(df.iloc[:, 7:13])
plt.figure(figsize=(8,8))
for binary in [True, False]:
    plt.scatter(feature[df['is_sllm'] == binary, 0], feature[df['is_sllm'] == binary, 1], alpha=0.8, label=binary)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc = 'best')
plt.grid()
plt.show()
```
![4b3563ff-d1db-470a-a000-a2d54d932509](https://github.com/atnikola/pokemon-analysis/assets/38530617/462add39-5b33-4c06-97d6-52ccd66b6996)


Although it's not 'perfect' we can clearly see that when the first principal component (PC1) reaches 50, we start to see a significantly higher concentration of legendary pokemon. Now, let's illustrate how much PC1 actually contributes to the explanatory variable (parameter) with a loading plot.

![6c28d244-5778-44f6-9485-84e38158cec2](https://github.com/atnikola/pokemon-analysis/assets/38530617/e4dd48ac-dd48-4c8a-928b-9c2d5f4af34d)


Assuming that PC1 is actually a strong indicator of whether or not a pokemon is classified as legendary, sub-legendary or mythical, it seems like Attack is one of the strongest indicators out of all stats (followed by Special Attack)

In PCA, we synthesized the "principal component" yPC1 which is a linear combination of the weight matrix (eigenvector) a for the explanatory variables. Here, define as many principal components as there are explanatory variables.
    
```python
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2, max_iter=500)
factors = fa.fit_transform(df.iloc[:, 7:13])
```

```python
plt.figure(figsize=(12, 12))
for binary in [True, False]:
    plt.scatter(factors[df['is_sllm'] == binary, 0], factors[df['is_sllm'] == binary, 1], alpha=0.8, label=binary)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.legend(loc = 'best')
plt.grid()
plt.show()
```
![87c2ebdc-0485-4696-add1-c58cfccaa03f](https://github.com/atnikola/pokemon-analysis/assets/38530617/94065410-00c5-42bf-bc8f-f4e75cd090f6)

In this instance, the determining factor of a 'legendary' is whether or not the sum of factor 1 and factor 2 exceeds a certain level, but it seems that it is slightly biased toward the larger factor 2. So which parameters do factor 2 and factor 1 allude to?

```python
plt.figure(figsize=(12, 12))
for x, y, name in zip(fa.components_[0], fa.components_[1], df.columns[7:13]):
    plt.text(x, y, name)
plt.scatter(fa.components_[0], fa.components_[1])
plt.grid()
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()
```
![9c22963c-da88-4a30-848e-5e90b10acc24](https://github.com/atnikola/pokemon-analysis/assets/38530617/a59e7c6d-f0ef-45f8-a78e-f617e4b61fb6)

Interestingly, it seems that BOTH factor 1 & 2 allude to DEFENSE

```python
X = df.iloc[:, 7:13]
y = df['Total']
```

```python
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Regression Coefficient= ", regr.coef_)
print("Intercept= ", regr.intercept_)
print("Coefficient of Determination= ", regr.score(X, y))
```
<img width="359" alt="Screenshot 2023-09-21 at 0 01 36" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/3b7294ef-b659-4799-8907-17feef51b3b4">

Let's see if we can predict the "Defense" stat with these 4 variables
```python
X = df.iloc[:, [7, 10, 11, 12]]
y = df['Defense']
```
```python
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Regression Coefficient= ", regr.coef_)
print("Intercept= ", regr.intercept_)
print("Coefficient of Determination= ", regr.score(X, y))
```
<img width="579" alt="Screenshot 2023-09-21 at 0 02 42" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/4ba91acb-c78e-451a-8713-27cc7b10bef3">

'Defense' = (0.16 * HP + -0.04 * Sp.Attack + 0.54 * Special Defense + -0.11 * Speed) + 33.4(intercept)

That's the relationship to predict Defense. Which makes sense for the most part as typically Special Defense correlates to Defense in games. HP being low/neutral is a bit surprising.
However, the coefficient of determination is so small that this may not be a reliable model..
    
# [Cross Validation & Regression Analysis](#cv-ra)
Since we saw earlier that Defense is a huge contributing factor to determining whether a pokemon is classified as 'legendary', let's use the rest of the stats to see if we can predict Defense and which stats are best contributors for determining this stat.
    
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)
```
    
```python
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print("Regression Coefficient= ", regr.coef_)
print("Intercept= ", regr.intercept_)
print("Coefficient of Determination(train)= ", regr.score(X_train, y_train))
print("Coefficient of Determination(test)= ", regr.score(X_test, y_test))
```
<img width="567" alt="Screenshot 2023-09-21 at 0 05 31" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/444b7c9a-2ad0-491a-8b6a-a9618b855415">
The above values change with each calculation because the division into training data and test data is random.
If you want to find a regression equation, you can do as above, but by standardizing the explanatory variables and objective variables and then regressing, you can find the "standard regression coefficient", which is an index of "importance of variables".

```python
Xs = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
ys = list(pd.DataFrame(y).apply(lambda x: (x-x.mean())/x.std()).values.reshape(len(y),))
```

```python
from sklearn import linear_model
regr = linear_model.LinearRegression()

regr.fit(Xs, ys)

print("Regression Coefficient= ", regr.coef_)
print("Intercept= ", regr.intercept_)
print("Coefficient of Determination= ", regr.score(Xs, ys))
```
<img width="574" alt="Screenshot 2023-09-21 at 0 06 50" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/d3da5b14-d23c-4fad-8bff-47328da9a771">

```python
pd.DataFrame(regr.coef_, index=list(df.columns[[7, 10, 11, 12]])).sort_values(0, ascending=False).style.bar(subset=[0])
```
<img width="220" alt="Screenshot 2023-09-21 at 0 07 11" src="https://github.com/atnikola/pokemon-analysis/assets/38530617/f0075a00-20cc-4aa7-8d8e-c2b7cbae44c6">

It seems that Special Defense is very important in predicting "Defense!!"

# [Conclusion](#conclusion)
It seems that Special Defense is the best determinator in predicting "Defense". This makes perfect sense as usually a High Defensive pokemon is usually also highly special defensive focused. Additionally, these pokemon are typically much slower which makes sense that 'Speed' would be negative. Additionally most 'non-defensive' pokemon could get 1HKO'ed by a strong move which is why Speed also plays a huge roll for those pokemon. The faster you are the higher the chance that you will attack first and survive. (Pretty basic for most Pokemon players >.>) 

Overall this was a way of exploring different pokemon traits and taking into account multiple factors. There's PLENTY more we can look into such as 'strengths', 'weaknesses' etc..which I may do at some point, but I hope you all enjoyed this, and thanks for reading all the way through! If you spot any errors or have any suggestions for visuals or further analysis, please feel free to drop a message!
