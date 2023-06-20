import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\crime_data.csv")
ewa.dtypes

# Exploratory Data Analysis
ewa.describe()
# Measures of Central Tendency / First moment business decision
ewa.Murder.mean()
ewa.Murder.median()
ewa.Murder.mode()

ewa.Assault.mean()
ewa.Assault.median()
ewa.Assault.mode()

ewa.UrbanPop.mean()
ewa.UrbanPop.median()
ewa.UrbanPop.mode()

ewa.Rape.mean()
ewa.Rape.median()
ewa.Rape.mode()

# Measures of Dispersion / Second moment business decision
ewa.Murder.var() # variance
ewa.Murder.std() # standard deviation

ewa.Assault.var() # variance
ewa.Assault.std() # standard deviation

ewa.UrbanPop.var() # variance
ewa.UrbanPop.std() # standard deviation

ewa.Rape.var() # variance
ewa.Rape.std() # standard deviation


# Third moment business decision
ewa.Murder.skew()
ewa.Assault.skew()
ewa.UrbanPop.skew()
ewa.Rape.skew()

# Fourth moment business decision
ewa.Murder.kurt()
ewa.Assault.kurt()
ewa.UrbanPop.kurt()
ewa.Rape.kurt()

# Data Visualization
import matplotlib.pyplot as plt

plt.hist(ewa.Murder) #histogram
plt.hist(ewa.Assault, color='red')
plt.hist(ewa.UrbanPop, color='black')
plt.hist(ewa.Rape, color='blue')

plt.boxplot(ewa.Murder) #boxplot
plt.boxplot(ewa.Assault) #boxplot
plt.boxplot(ewa.UrbanPop) #boxplot
plt.boxplot(ewa.Rape) #boxplot

_______________________________________________________________________________
# outlier treatment
import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\crime_data.csv")
ewa.dtypes

# let's find outliers 

sns.boxplot(ewa.Murder) # no outliers
sns.boxplot(ewa.Assault) # no outliers
sns.boxplot(ewa.UrbanPop) # no outliers
sns.boxplot(ewa.Rape)

# So, we have 1 variables which has outliers
________________________________________________________________________________
# 1. Rape
# Detection of outliers 
IQR = ewa['Rape'].quantile(0.75) - ewa['Rape'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Rape'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Rape'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Rape'] > upper_limit, True, np.where(ewa['Rape'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Rape)
# we see no outiers
________________________________________________________________________________
############### Now Our data is Outlier free ###############
__________________________________________________________
# zero variance and near zero variance 

# If the variance is low or close to zero, then a feature is approximately 
# constant and will not improve the performance of the model.
# In that case, it should be removed. 

ewa.var() # variance of numeric variables
ewa.var() == 0
_______________________________________________________________________________
# Missing Values Imputation 
import numpy as np
import pandas as pd
# check for count of NA'sin each column
ewa.isna().sum()
# there are no missing values
_______________________________________________________________________________
### Identify duplicates records in the data ###

ewa = ewa.duplicated()
ewa
sum(ewa)

# we have no duplicate records in the data.
# Removing Duplicates
# ewa1 = ewa.drop_duplicates()
_______________________________________________________________________________
## Standardization and Normalization #########
import pandas as pd
import numpy as np

### Standardization
from sklearn.preprocessing import StandardScaler
d = pd.read_csv(r"C:\3. Data preprocessing\DataSets-Data Pre Processing\DataSets\Seeds_data.csv")

a = d.describe()
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
df = scaler.fit_transform(a)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()


### Normalization
## load data set
ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\crime_data.csv")
ewa.columns

a1 = ewa.describe()

# get dummies
ewa = pd.get_dummies(ewa, drop_first = True)

### Normalization function - Custom Function
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

ewa_norm = norm_func(ewa)
b = ewa_norm.describe()
___________________________________________________________
# 1st of all import all the packages
import pandas as pd
import matplotlib.pylab as plt
# Load the data
ewa1 = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\crime_data.csv")

ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["Unnamed: 0"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(ewa.iloc[:, 0:])
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

ewa['clust'] = cluster_labels # creating a new column and assigning it to new column 

ewa1 = ewa.iloc[:, [4,0,1,2,3]]
ewa1.head()

# Aggregate mean of each cluster
ewa1.iloc[:, 1:].groupby(ewa1.clust).mean()
ewa1.iloc[:, 1:].groupby(ewa1.clust).std()

# creating a csv file 
ewa1.to_csv("crime_data.csv", encoding = "utf-8")


import os
os.getcwd()