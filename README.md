# Pokémon Analysis
Andreas Nikolaidis
February 2022 - Jupyter Notebook (Link)

- [Introduction](docs/README.md/#Introduction)
- [Exploratory Analysis](docs/README.md)
- [Correlations & Descriptive Statistics](docs/README.md)
- [Principal Component Analysis (PCA)](docs/README.md)
- [Cross Validation & Regression Analysis](docs/README.md)
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

Read Data File:
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

Let's create a diverging bar to determine the rate at which legendary pokemon appear in each generation:
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

## Correlations & Descriptive Statistics 
Let's move to explore some correlations between stats.

```python
#Correlation
Base_stats = ['Primary','Secondary','Classification','%Male','%Female',
              'Height','Weight','Capture_Rate','Base_Steps','HP','Attack','Defense',
              'SP_Attack','SP_Defense','Speed','is_sllm']

df_BS = df[Base_stats]
df_BS.head()
```
```python
plt.figure(figsize=(14,12))

heatmap = sns.heatmap(df_BS.corr(), vmin=-1,vmax=1, annot=True, cmap='Blues')

heatmap.set_title('Correlation Base Stats Heatmap', fontdict={'fontsize':15}, pad=12)
plt.show()
```
![correlation_plot](https://user-images.githubusercontent.com/38530617/153744026-f4ad82be-09c4-4cc7-bbeb-36571a98e397.png)


```python
p1 = sns.jointplot(x="SP_Attack",y="SP_Defense",data=df,kind="hex",color="lightgreen")
p1.fig.suptitle("Hex Plot of Special Attack and Special Defense - Some Correlation")
p1.fig.subplots_adjust(top=0.95)
p2 = sns.jointplot(x="Defense",y="SP_Defense",data=df,kind="hex",color="lightblue")
p2.fig.suptitle("Hex Plot of Defense and Special Defense - Some Correlation")
p2.fig.subplots_adjust(top=0.95)
p3 = sns.jointplot(x="SP_Attack",y="Speed",data=df,kind="hex",color="pink")
p3.fig.suptitle("Hex Plot of Special Attack and Speed - Some Correlation")
p3.fig.subplots_adjust(top=0.95)
p4 = sns.jointplot(x="Attack",y="SP_Attack",data=df,kind="hex",color="orange")
p4.fig.suptitle("Hex Plot of Attack and Special Attack - Some Correlation")
p4.fig.subplots_adjust(top=0.95)
p5 = sns.jointplot(x="Attack",y="Defense",data=df,kind="hex",color="purple")
p5.fig.suptitle("Hex Plot of Attack and Defense - Some Correlation")
p5.fig.subplots_adjust(top=0.95)
```
![hex_green](https://user-images.githubusercontent.com/38530617/153745763-7d378c6e-efbd-4afa-a852-2fe983ffae76.png)
![hex_blue](https://user-images.githubusercontent.com/38530617/153745765-d36af02d-afc1-4226-8586-df5301d450eb.png)
![hex_red](https://user-images.githubusercontent.com/38530617/153745775-e0c6e828-d52b-4758-9406-ced67a9d36b6.png)
![hex_orange](https://user-images.githubusercontent.com/38530617/153745779-f30e6bfc-8a29-4989-822c-cb604bbd0e98.png)
![hex_purple](https://user-images.githubusercontent.com/38530617/153745767-9e35d803-87a0-488b-b540-72beaeb09104.png)

```python
from pandas import plotting
type1 = list(set(list(df['Primary'])))
cmap = plt.get_cmap('viridis')
colors = [cmap((type1.index(c) + 1) / (len(type1) + 2)) for c in df['Primary'].tolist()]
plotting.scatter_matrix(df.iloc[:, 13:18], figsize=(15, 15), color=colors, alpha=0.7) 
plt.show()
```
![corrplot](https://user-images.githubusercontent.com/38530617/153744465-059f49d5-697e-43b4-b25d-b1ebdb409eba.png)

```python
import numpy as np
pd.DataFrame(np.corrcoef(df.iloc[:, 13:18].T.values.tolist()), 
             columns=df.iloc[:, 13:18].columns, index=df.iloc[:, 13:18].columns)
```
<img width="477" alt="corrplot values" src="https://user-images.githubusercontent.com/38530617/153744483-6d01ea6a-b457-4ccc-94d0-4fc2f00118cb.png">

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
![attack_defense](https://user-images.githubusercontent.com/38530617/153744067-1e1730c9-c360-483d-9159-45d9b53f452b.png)

```python
fig, ax = plt.subplots(2, 3, figsize=(14, 8), sharey=True)

spines = ["top","right","left"]
for i, col in enumerate(["HP", "Attack", "Defense", "SP_Attack", "SP_Defense", "Speed"]):
    sns.kdeplot(x=col, data=df, label=col, ax=ax[i//3][i%3],
                fill=True, color='lightblue', linewidth=2
               )
    
    ax[i//3][i%3].set_xlim(-5, 250)
    
    for s in spines:
        ax[i//3][i%3].spines[s].set_visible(False)
        

plt.tight_layout()
plt.show()
```
![density_plots](https://user-images.githubusercontent.com/38530617/153744330-f743c4ac-a7c6-4384-a190-b83eb69c19a2.png)

```python
df.describe()
```
<img width="973" alt="std_dev_att_def" src="https://user-images.githubusercontent.com/38530617/153744167-f8896e9f-2a77-486b-a409-36f9bcc85d55.png">

Looking at the summary statistics, we can see that the assumption about the variance and skewness of both plots was correct. The ‘std’ metric of the Attack is less than Defense, meaning that Defense statistics are more spread. Similarly, the Sp.Atk ‘std’ is larger than that of the Sp.Def. Skewness is determined by the positions of the median (50%) and the mean. Since in all instances (Attack, Defense, Sp.Attack and Sp.Defense) the mean is greater than the median, it is emphasised that the distribution is right-skewed (positively skewed).

## Principal Component Analysis (PCA)
Let's analyze 800+ Pokemon as principal components and plot them in a two-dimensional plane using the first and second principal components.
Principal component analysis (PCA) is a type of multivariate analysis method that is often used as a dimensionality reduction method.

In this data, the characteristics of 800+ Pokemon are represented by **6 types** of **"observed variables"** (x1, x2, x3, x4, x5, x6). 
These 6 variables are used as explanatory variables. 
On the other hand, the synthetic variable synthesized by PCA is called "principal component score" and is given by a linear combination as shown in the following equation: 
<INSERT IMG>


In principal component analysis, the larger the ```eigenvalue``` (= variance of the principal component score), the **more important the principal component score is**.
PCA is also sometimes regarded as a type of "unsupervised machine learning" and reveals the structure of the data itself. So let's start by importing ```PCA``` from ```Scikit-learn```

```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df.iloc[:, 13:18])
feature = pca.transform(df.iloc[:, 13:18])
plt.figure(figsize=(15, 15))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
plt.show()
```
![PCA](https://user-images.githubusercontent.com/38530617/153744763-f3c32c09-0534-4c58-809f-e0212e10ca05.png)

```python
import matplotlib.ticker as ticker
import numpy as np
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()
```
![components](https://user-images.githubusercontent.com/38530617/153744830-3f86c909-c3de-4ff1-bbae-c0c32cd90f53.png)
    
Let's see if we can determine what makes a 'legendary' pokemon

```python
pca = PCA()
pca.fit(df.iloc[:, 13:18])
feature = pca.transform(df.iloc[:, 13:18])
plt.figure(figsize=(15, 15))
for binary in [True, False]:
    plt.scatter(feature[df['is_sllm'] == binary, 0], feature[df['is_sllm'] == binary, 1], alpha=0.8, label=binary)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend(loc = 'best')
plt.grid()
plt.show()
```
![pca_color](https://user-images.githubusercontent.com/38530617/153744885-42ed952b-05de-4f8a-a658-b6b64ba1d29d.png)

Nice! Although it's not 'exact' we can clearly see that when the first principal component (PC1) reaches 50, we start to see a significantly higher concentration of legendary pokemon! Now, let's illustrate how much PC1 actually contributes to the explanatory variable (parameter) with a loading plot.

![components_stats](https://user-images.githubusercontent.com/38530617/153745198-40a1fa84-cb1a-4118-a0d1-68ed7a50bb84.png)

Assuming that the first principal component (PC1) is actually a strong indicator of whether or not a pokemon is classified as legendary, sub-legendary or mythical, it seems like Special Attack is the best indicator out of all stats (follow by Physical Attack)

In the second principal component (PC2), Defense and Speed contribute to the opposite: Positive & Negative. 

"Factor Analysis" is a method that is similar to principal component analysis.

In PCA, we synthesized the "principal component" yPC1 which is a linear combination of the weight matrix (eigenvector) a for the explanatory variables. Here, define as many principal components as there are explanatory variables.

yPC1 = a1,1 x1 + a1,2 x2 + a1,3 x3 + a1,4 x4 + a1,5 + ...

In factor analysis, based on the idea that the explanatory variable (observed variable) x is synthesized from a latent variable called "factor", the factor score f, the weight matrix (factor load) w, and the unique factor e are specified. (There is no idea of ​​a unique factor in principal component analysis).

x1 = w1,1 f1 + w1,2 f2 + e1

x2 = w2,1 f1 + w2,2 f2 + e2

x3 = w3,1 f1 + w3,2 f2 + e3

x4 = w4,1 f1 + w4,2 f2 + e4

x5 = w5,1 f1 + w5,2 f2 + e5

x6 = w6,1 f1 + w6,2 f2 + e6

The factor score f is a latent variable unique to each individual (sample). The linear sum of the factor score and the factor load (w1,1 f1 + w1,2 f2, etc.) is called the "common factor" and can be observed as an "observed variable" by adding it to the "unique factor" e unique to the observed variable. It's a way of thinking. The number of factors is usually smaller than the explanatory variables and must be decided in advance.

(However, terms such as common factors and factors are very confusing because it seems that different people have different definitions as far as I can see)
    
```python
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2, max_iter=500)
factors = fa.fit_transform(df.iloc[:, 13:18])
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
![pca_color2](https://user-images.githubusercontent.com/38530617/153745352-7366b274-3ff8-4bbc-bd4d-327bec30f3eb.png)

In this instance, the determining factor of a 'legendary' is whether or not the sum of factor 1 and factor 2 exceeds a certain level, but it seems that it is slightly biased toward the larger factor 2. So which parameters do factor 2 and factor 1 allude to?

```python
plt.figure(figsize=(8, 8))
for x, y, name in zip(fa.components_[0], fa.components_[1], df.columns[13:18]):
    plt.text(x, y, name)
plt.scatter(fa.components_[0], fa.components_[1])
plt.grid()
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.show()
```
![component_stats(factor2)](https://user-images.githubusercontent.com/38530617/153745455-4f348603-8330-4d05-a467-261ebc35c02c.png)

Factor 1 highest value = "Defense"
Factor 2 highest value = "Special Attack" 

Let's create some charts!

Firstly I created a dendrogram (dendro = greek word for tree :)) for all pokemon (Image file is way too large to display clearly)
```python
dfs = df.iloc[:, 13:18].apply(lambda x: (x-x.mean())/x.std(), axis=0)
from scipy.cluster.hierarchy import linkage, dendrogram
result1 = linkage(dfs, 
                  metric = 'euclidean', 
                  method = 'average')
plt.figure(figsize=(15, 150))
dendrogram(result1, orientation='right', labels=list(df['Name']), color_threshold=2)
plt.title("Dedrogram of Pokemon")
plt.xlabel("Threshold")
plt.grid()
plt.show()
```
```python
def get_cluster_by_number(result, number):
    output_clusters = []
    x_result, y_result = result.shape
    n_clusters = x_result + 1
    cluster_id = x_result + 1
    father_of = {}
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(len(result) - 1):
        n1 = int(result[i][0])
        n2 = int(result[i][1])
        val = result[i][2]
        n_clusters -= 1
        if n_clusters >= number:
            father_of[n1] = cluster_id
            father_of[n2] = cluster_id

        cluster_id += 1

    cluster_dict = {}
    for n in range(x_result + 1):
        if n not in father_of:
            output_clusters.append([n])
            continue

        n2 = n
        m = False
        while n2 in father_of:
            m = father_of[n2]
            #print [n2, m]
            n2 = m

        if m not in cluster_dict:
            cluster_dict.update({m:[]})
        cluster_dict[m].append(n)

    output_clusters += cluster_dict.values()

    output_cluster_id = 0
    output_cluster_ids = [0] * (x_result + 1)
    for cluster in sorted(output_clusters):
        for i in cluster:
            output_cluster_ids[i] = output_cluster_id
        output_cluster_id += 1

    return output_cluster_ids
```
```python
clusterIDs = get_cluster_by_number(result1, 50)
print(clusterIDs)
```
<img width="1364" alt="cluster_ids" src="https://user-images.githubusercontent.com/38530617/153746879-35524b22-aa1a-46a0-b807-9712d474bda0.png">

```python
plt.hist(clusterIDs, bins=50)
plt.show()
```
![histo](https://user-images.githubusercontent.com/38530617/153746942-f5986c3e-0267-413d-80bc-b744e84b46cd.png)

Here we've created a histogram of clusters of pokemon that exhibit similar traits with each other. Here we've created 50 bins so there will be 50 different clusters of pokemon. That's quite a large number of charts to display so I'll just display several so you get the idea.

![cluster4](https://user-images.githubusercontent.com/38530617/153747188-3ea042df-70b5-4972-bf44-4e3b9178be93.png)
![cluster5](https://user-images.githubusercontent.com/38530617/153747187-01251b73-257b-4fe6-a093-2c8423980726.png)
![cluster6](https://user-images.githubusercontent.com/38530617/153747185-05c2cf18-828c-45d2-9236-a2dad7dbf9c3.png)
![cluster8](https://user-images.githubusercontent.com/38530617/153747184-043fe9bf-bfa7-4b48-ab16-24668a4a3f99.png)
![cluster10](https://user-images.githubusercontent.com/38530617/153747182-e0dcd7e9-5bfc-4d70-8706-0e02c768f361.png)
![cluster50](https://user-images.githubusercontent.com/38530617/153747180-f55a5feb-fab9-4753-a102-433b3cc19411.png)


Some pokemon exhibit lots of traits similar to each other while others (like Regieleki) stand out.
    
# Cross Validation
Since we saw earlier that Special Attack is a huge contributing factor to determining whether a pokemon is classified as 'legendary', let's use the rest of the stats to see if we can predict Special Attack!
    
```python
X = df.iloc[:, 13:18]
y = df['total']
```
    
```python
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

print("Regression Coefficient= ", regr.coef_)
print("Intercept= ", regr.intercept_)
print("Coefficient of Determination= ", regr.score(X, y))
```

```python
df.columns[[12, 13, 14, 16, 17]]
X = df.iloc[:, [12, 13, 14, 16, 17]]
y = df['SP_Attack']
```
### Cross Validation
In machine learning, in order to evaluate performance, known data is divided into ```training``` and ```test``` data. Training (learning) is performed using training data to build a prediction model, and performance evaluation is performed based on how accurately the test data that was not used to build the prediction model can be predicted. Such an evaluation method is called "cross-validation".

Training data (60% of all data)
X_train: Explanatory variables for training data
y_train: Objective variable for training data
Test data (40% of all data)
X_test: Explanatory variable for test data
y_test: Objective variable for test data
We aim to learn the relationship between X_train and y_train and predict y_test from X_test. If the training data seems to show good performance, but the test data not used for training has poor performance, the model is said to be "overfitted".
    
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

```
Regression Coefficient=  [ 0.15598049  0.09796333 -0.11115187  0.47986509  0.32513351]
Intercept=  5.4684249031776915
Coefficient of Determination(train)=  0.39594357153305826
Coefficient of Determination(test)=  0.38127048972638855
```
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
```
Regression Coefficient=  [ 0.152545    0.11255532 -0.09718819  0.40725508  0.28208903]
Intercept=  1.1730486200365748e-16
Coefficient of Determination=  0.3958130072204933
```
```
pd.DataFrame(regr.coef_, index=list(df.columns[[12, 13, 14, 16, 17]])).sort_values(0, ascending=False).style.bar(subset=[0])
```
<img width="224" alt="sp attack prediction" src="https://user-images.githubusercontent.com/38530617/153748185-68eac678-43dc-4636-98fd-ec6ed14bd448.png">

It seems that Special Defense & Speed are very important in predicting "Special Attack"
    
# Conclusion
Regression analysis, such as multiple regression analysis, uses numerical data as an explanatory variable and predicts numerical data as an objective variable. On the other hand, quantification type I predicts using non-numeric categorical data as an explanatory variable and numerical data as an objective variable. When the explanatory variables are a mixture of numerical data and categorical data, they are called extended quantification type I.

Overall this was a way of exploring different pokemon traits and taking into account multiple factors. There's plenty more we can look into such as 'strengths', 'weaknesses' etc.. I hope you all enjoyed this, and thanks for reading all the way through!
