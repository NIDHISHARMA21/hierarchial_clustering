import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\EastWestAirlines.xlsx",sheet_name="data")
ewa.dtypes
ewa.describe()
# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
ewa.Balance.mean() 
ewa.Balance.median()
ewa.Balance.mode()

ewa.Qual_miles.mean()
ewa.Qual_miles.median()
ewa.Qual_miles.mode()

ewa.cc1_miles.mean() 
ewa.cc1_miles.median()
ewa.cc1_miles.mode()

ewa.cc2_miles.mean()
ewa.cc2_miles.median()
ewa.cc2_miles.mode()

ewa.cc3_miles.mean()
ewa.cc3_miles.median()
ewa.cc3_miles.mode()

ewa.Bonus_miles.mean()
ewa.Bonus_miles.median()
ewa.Bonus_miles.mode()

ewa.Bonus_trans.mean()
ewa.Bonus_trans.median()
ewa.Bonus_trans.mode()

ewa.Flight_miles_12mo.mean()
ewa.Flight_miles_12mo.median()
ewa.Flight_miles_12mo.mode()

ewa.Flight_trans_12.mean()
ewa.Flight_trans_12.median()
ewa.Flight_trans_12.mode()

ewa.Days_since_enroll.mean()
ewa.Days_since_enroll.median()
ewa.Days_since_enroll.mode()

ewa.Award.mean() 
ewa.Award.median()
ewa.Award.mode()

ewa.dtypes
# Measures of Dispersion / Second moment business decision
ewa.Balance.var() # variance
ewa.Balance.std() # standard deviation

ewa.Qual_miles.var() # variance
ewa.Qual_miles.std() # standard deviation

ewa.cc1_miles.var() # variance
ewa.cc1_miles.std() # standard deviation

ewa.cc2_miles.var() # variance
ewa.cc2_miles.std() # standard deviation

ewa.cc3_miles.var() # variance
ewa.cc3_miles.std() # standard deviation

ewa.Bonus_miles.var() # variance
ewa.Bonus_miles.std() # standard deviation

ewa.Bonus_trans.var() # variance
ewa.Bonus_trans.std() # standard deviation

ewa.Flight_miles_12mo.var() # variance
ewa.Flight_miles_12mo.std() # standard deviation

ewa.Flight_trans_12.var() # variance
ewa.Flight_trans_12.std() # standard deviation

ewa.Days_since_enroll.var() # variance
ewa.Days_since_enroll.std() # standard deviation

ewa.Award.var() # variance
ewa.Award.std() # standard deviation


# Third moment business decision
ewa.Balance.skew()
ewa.Qual_miles.skew()
ewa.cc1_miles.skew()
ewa.cc2_miles.skew()
ewa.cc3_miles.skew()
ewa.Bonus_miles.skew()
ewa.Bonus_trans.skew()
ewa.Flight_miles_12mo.skew()
ewa.Flight_trans_12.skew()
ewa.Days_since_enroll.skew()
ewa.Award.skew()


# Fourth moment business decision
ewa.Balance.kurt()
ewa.Qual_miles.kurt()
ewa.cc1_miles.kurt()
ewa.cc2_miles.kurt()
ewa.cc3_miles.kurt()
ewa.Bonus_miles.kurt()
ewa.Bonus_trans.kurt()
ewa.Flight_miles_12mo.kurt()
ewa.Flight_trans_12.kurt()
ewa.Days_since_enroll.kurt()
ewa.Award.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

 #boxplot
plt.boxplot(ewa.Balance)
plt.boxplot(ewa.Qual_miles)
plt.boxplot(ewa.cc1_miles)
plt.boxplot(ewa.cc2_miles)
plt.boxplot(ewa.cc3_miles)
plt.boxplot(ewa.Bonus_miles)
plt.boxplot(ewa.Bonus_trans)
plt.boxplot(ewa.Flight_miles_12mo)
plt.boxplot(ewa.Flight_trans_12)
plt.boxplot(ewa.Days_since_enroll)

 #histogram
plt.hist(ewa.Balance)
plt.hist(ewa.Qual_miles, color='red')
plt.hist(ewa.cc1_miles, color='black')
plt.hist(ewa.cc2_miles, color='blue')
plt.hist(ewa.cc3_miles, color='yellow')
plt.hist(ewa.Bonus_miles, color='orange')
plt.hist(ewa.Bonus_trans, color='pink')
plt.hist(ewa.Flight_miles_12mo, color='indigo')
plt.hist(ewa.Flight_trans_12, color='violet')
plt.hist(ewa.Days_since_enroll, color='green')
plt.hist(ewa.Award, color='red')

_______________________________________________________________________________
# outlier treatment
# let's find outliers 

sns.boxplot(ewa.Balance)
sns.boxplot(ewa.Qual_miles)
sns.boxplot(ewa.cc1_miles) # no outliers
sns.boxplot(ewa.cc2_miles)
sns.boxplot(ewa.cc3_miles)
sns.boxplot(ewa.Bonus_miles)
sns.boxplot(ewa.Bonus_trans)
sns.boxplot(ewa.Flight_miles_12mo)
sns.boxplot(ewa.Flight_trans_12)
sns.boxplot(ewa.Days_since_enroll) # no outliers

# So, we have 8 variables which has outliers
ewa.dtypes
________________________________________________________________________________
# 1. Balance
# Detection of outliers 
IQR = ewa['Balance'].quantile(0.75) - ewa['Balance'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Balance'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Balance'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Balance'])
df_t = winsor.fit_transform(ewa[['Balance']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Balance)# we see no outiers
________________________________________________________________________________
# 2. Qual_miles
# Detection of outliers 
IQR = ewa['Qual_miles'].quantile(0.75) - ewa['Qual_miles'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Qual_miles'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Qual_miles'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Qual_miles'])

df_t = winsor.fit_transform(ewa[['Qual_miles']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Qual_miles)# we see no outiers

________________________________________________________________________________
# 3. cc2_miles
# Detection of outliers 
IQR = ewa['cc2_miles'].quantile(0.75) - ewa['cc2_miles'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['cc2_miles'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['cc2_miles'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['cc2_miles'])

df_t = winsor.fit_transform(ewa[['cc2_miles']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.cc2_miles)# we see no outiers
________________________________________________________________________________
# 4. cc3_miles
# Detection of outliers 
IQR = ewa['cc3_miles'].quantile(0.75) - ewa['cc3_miles'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['cc3_miles'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['cc3_miles'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['cc3_miles'])

df_t = winsor.fit_transform(ewa[['cc3_miles']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.cc3_miles)# we see no outiers

________________________________________________________________________________
# 5. Bonus_miles
# Detection of outliers 
IQR = ewa['Bonus_miles'].quantile(0.75) - ewa['Bonus_miles'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Bonus_miles'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Bonus_miles'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_miles'])

df_t = winsor.fit_transform(ewa[['Bonus_miles']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Bonus_miles)# we see no outiers
________________________________________________________________________________
# 6. Bonus_trans
# Detection of outliers 
IQR = ewa['Bonus_trans'].quantile(0.75) - ewa['Bonus_trans'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Bonus_trans'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Bonus_trans'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Bonus_trans'])

df_t = winsor.fit_transform(ewa[['Bonus_trans']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Bonus_trans)# we see no outiers
________________________________________________________________________________
# 7. Flight_miles_12mo
# Detection of outliers 
IQR = ewa['Flight_miles_12mo'].quantile(0.75) - ewa['Flight_miles_12mo'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Flight_miles_12mo'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Flight_miles_12mo'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Flight_miles_12mo'])

df_t = winsor.fit_transform(ewa[['Flight_miles_12mo']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Flight_miles_12mo)# we see no outiers
________________________________________________________________________________
# 8. Flight_trans_12
# Detection of outliers 
IQR = ewa['Flight_trans_12'].quantile(0.75) - ewa['Flight_trans_12'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Flight_trans_12'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Flight_trans_12'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 3. Winsorization ###############
# pip install feature_engine   # install the package
from feature_engine.outliers import Winsorizer
winsor = Winsorizer(capping_method='iqr', # choose  IQR rule boundaries or gaussian for mean and std
                          tail='both', # cap left, right or both tails 
                          fold=1.5,
                          variables=['Flight_trans_12'])
df_t = winsor.fit_transform(ewa[['Flight_trans_12']])

# we can inspect the minimum caps and maximum caps 
# winsor.left_tail_caps_, winsor.right_tail_caps_

# lets see boxplot
sns.boxplot(df_t.Flight_trans_12)# we see no outiers
________________________________________________________________________________
############### Now Our data is Outlier free ###############
__________________________________________________________
# zero variance and near zero variance 

ewa.var() # variance of numeric variables
ewa.var() == 0
_______________________________________________________________________________
# Missing Values Imputation 
ewa.isna().sum()
# there are no missing values
_______________________________________________________________________________
### Identify duplicates records in the data ###
ewa1 = ewa.duplicated()
ewa1
sum(ewa1)
# we have no duplicate records in the data.
_______________________________________________________________________________
### Normalization
# Range converts to: 0 to 1
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)
___________________________________________________________
ewa = ewa.drop(["ID#"], axis=1)

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

ewa1 = ewa.iloc[:, [11,0,1,2,3,4,5,6,7,8,9,10]]
ewa1.head()

# Aggregate mean of each cluster
ewa1.iloc[:, 1:].groupby(ewa1.clust).mean()
ewa1.iloc[:, 1:].groupby(ewa1.clust).std()

# creating a csv file 
ewa1.to_csv("EastWestAirlines.csv", encoding = "utf-8")


import os
os.getcwd()