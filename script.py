#import necessary packages
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


#Read Data File:
df = pd.read_excel("pokemon.xlsx")

#Create a separate dataframe including just the necessary stats:
df_stats = df[["Name","HP","Attack","Defense","SP_Attack","SP_Defense","Speed"]]
df['total'] = df.HP + df.Attack + df.Defense + df.SP_Attack + df.SP_Defense + df.Speed



#palette: https://seaborn.pydata.org/tutorial/color_palettes.html?highlight=color
plt.figure(figsize=(13,10), dpi=80)
sns.violinplot(x='Gen', y='total', data=df, scale='width', inner='quartile', palette='Set2') 
plt.title('Violin Plot of Total Stats by Generation', fontsize=22)

plt.figure(figsize=(13,10), dpi=80)
sns.violinplot(x='Gen', y='Speed', data=df, scale='width', inner='quartile', palette='Set2')
plt.title('Violin Plot of Total Stats by Generation', fontsize=22)

def top_n(df, category, n):
    return (df.loc[df['Gen'] == 'VIII'].sort_values(category, ascending=False)[['Name','Gen',category]].head(n))


top_n(df, 'Speed', 10)

types_color_dict = {
    'grass':'#8ED752', 'fire':'#F95643', 'water':'#53AFFE', 'bug':"#C3D221", 'normal':"#BBBDAF", \
    'poison': "#AD5CA2", 'electric':"#F8E64E", 'ground':"#F0CA42", 'fairy':"#F9AEFE", \
    'fighting':"#A35449", 'psychic':"#FB61B4", 'rock':"#CDBD72", 'ghost':"#7673DA", \
    'ice':"#66EBFF", 'dragon':"#8B76FF", 'dark':"#1A1A1A", 'steel':"#C3C1D7", 'flying':"#75A4F9" }

plt.figure(figsize=(15,12), dpi=80)
sns.violinplot(x='Primary', y='total', data=df, scale='width', inner='quartile', palette=types_color_dict)

plt.title('Violin Plot of Total Stats by Type', fontsize=20)

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

labels = ['Mono type pokemon', 'Dual type pokemon']
sizes = [monotype, dualtype]
colors = ['lightskyblue', 'lightcoral']

patches, texts, _ = plt.pie(sizes, colors=colors, autopct='%1.1f%%', startangle=90, explode=(0,0.1))
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.title('Dual-Type Ratio', fontsize=12)
plt.tight_layout()
plt.show()

#Sub-Legendary, Legendary or Mythical:
df.loc[df["is_sllm"]==False,"sllmid"] = 0
df.loc[df["is_sllm"]==True,"sllmid"] = 1


sllm_ratio = df.groupby("Gen").mean()["sllmid"]
sllm_ratio.round(4)*100

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
#plt.show()


#Correlation
Base_stats = ['Primary','Secondary','Classification','%Male','%Female',
              'Height','Weight','Capture_Rate','Base_Steps','HP','Attack','Defense',
              'SP_Attack','SP_Defense','Speed','is_sllm']

df_BS = df[Base_stats]
df_BS.head()

plt.figure(figsize=(14,12))

heatmap = sns.heatmap(df_BS.corr(), vmin=-1,vmax=1, annot=True, cmap='Blues')

heatmap.set_title('Correlation Base Stats Heatmap', fontdict={'fontsize':15}, pad=12)


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

#from pandas import plotting
type1 = list(set(list(df['Primary'])))
cmap = plt.get_cmap('viridis')
colors = [cmap((type1.index(c) + 1) / (len(type1) + 2)) for c in df['Primary'].tolist()]
plotting.scatter_matrix(df.iloc[:, 13:18], figsize=(15, 15), color=colors, alpha=0.7) 

#import numpy as np
pd.DataFrame(np.corrcoef(df.iloc[:, 13:18].T.values.tolist()), 
             columns=df.iloc[:, 13:18].columns, index=df.iloc[:, 13:18].columns)

labels = ["Defense", "Attack"]
dims = (11.7, 8.27) #a4
fig, ax = plt.subplots(figsize=dims)
Defhist = sns.distplot(df['Defense'],color='g', hist=True, ax=ax)
Atthist = sns.distplot(df['Attack'],color='r', hist=True, ax=ax)
Atthist.set(title='Distribution of Defense & Attack')
plt.legend(labels, loc="best")
FigHist = Atthist.get_figure()

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
#plt.show()

#Scikit-learn
#from sklearn.decomposition import PCA
pca = PCA()
pca.fit(df.iloc[:, 13:18])
feature = pca.transform(df.iloc[:, 13:18])
plt.figure(figsize=(15, 15))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.grid()
#plt.show()

#import matplotlib.ticker as ticker
#import numpy as np
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution ratio")
plt.grid()
plt.show()

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
#plt.show()

#from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=2, max_iter=500)
factors = fa.fit_transform(df.iloc[:, 13:18])


plt.figure(figsize=(12, 12))
for binary in [True, False]:
    plt.scatter(factors[df['is_sllm'] == binary, 0], factors[df['is_sllm'] == binary, 1], alpha=0.8, label=binary)
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
plt.legend(loc = 'best')
plt.grid()
#plt.show()

plt.figure(figsize=(8, 8))
for x, y, name in zip(fa.components_[0], fa.components_[1], df.columns[13:18]):
    plt.text(x, y, name)
plt.scatter(fa.components_[0], fa.components_[1])
plt.grid()
plt.xlabel("Factor 1")
plt.ylabel("Factor 2")
#plt.show()

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
#plt.show()

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

clusterIDs = get_cluster_by_number(result1, 50)
#print(clusterIDs)

plt.hist(clusterIDs, bins=50)
#plt.show()

# Cross Validation

X = df.iloc[:, 13:18]
y = df['total']
    

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X, y)

#print("Regression Coefficient= ", regr.coef_)
#print("Intercept= ", regr.intercept_)
#print("Coefficient of Determination= ", regr.score(X, y))


df.columns[[12, 13, 14, 16, 17]]
X = df.iloc[:, [12, 13, 14, 16, 17]]
y = df['SP_Attack']


#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

#from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
#print("Regression Coefficient= ", regr.coef_)
#print("Intercept= ", regr.intercept_)
#print("Coefficient of Determination(train)= ", regr.score(X_train, y_train))
#print("Coefficient of Determination(test)= ", regr.score(X_test, y_test))


Xs = X.apply(lambda x: (x-x.mean())/x.std(), axis=0)
ys = list(pd.DataFrame(y).apply(lambda x: (x-x.mean())/x.std()).values.reshape(len(y),))

#from sklearn import linear_model
regr = linear_model.LinearRegression()

regr.fit(Xs, ys)

#print("Regression Coefficient= ", regr.coef_)
#print("Intercept= ", regr.intercept_)
#print("Coefficient of Determination= ", regr.score(Xs, ys))


pd.DataFrame(regr.coef_, index=list(df.columns[[12, 13, 14, 16, 17]])).sort_values(0, ascending=False).style.bar(subset=[0])