import pandas as pd
import matplotlib.pylab as plt

auto = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")

auto.describe()
auto.info()

autonew = auto.drop(["Customer","State","Effective To Date"], axis=1)
#create dummy variabls
df_new=pd.get_dummies(autonew)
df_new_1=pd.get_dummies(autonew,drop_first=True)
# we have created dummies for all categorical columns
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df_new_1.iloc[:, 0:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

auto['clust'] = cluster_labels # creating a new column and assigning it to new column 

auto = auto.iloc[:, [24,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]
auto.head()

# inferences
# count the number in each cluster
auto['clust'].value_counts()
# Aggregate mean of each cluster
auto.iloc[:, 3:].groupby(auto.clust).mean()
