import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
ewa.dtypes
# Exploratory Data Analysis
ewa.describe()
# Measures of Central Tendency / First moment business decision
ewa.Customer_Lifetime_Value.mean()
ewa.Customer_Lifetime_Value.median()
ewa.Customer_Lifetime_Value.mode()

ewa.Income.mean()
ewa.Income.median()
ewa.Income.mode()

ewa.Monthly_Premium_Auto.mean()
ewa.Monthly_Premium_Auto.median()
ewa.Monthly_Premium_Auto.mode()

ewa.Months_Since_Last_Claim .mean()
ewa.Months_Since_Last_Claim .median()
ewa.Months_Since_Last_Claim .mode()

ewa.Months_Since_Policy_Inception.mean() 
ewa.Months_Since_Policy_Inception.median()
ewa.Months_Since_Policy_Inception.mode()

ewa.Number_of_Open_Complaints.mean() 
ewa.Number_of_Open_Complaints.median()
ewa.Number_of_Open_Complaints.mode()

ewa.Number_of_Policies.mean()
ewa.Number_of_Policies.median()
ewa.Number_of_Policies.mode()

ewa.Total_Claim_Amount.mean()
ewa.Total_Claim_Amount.median()
ewa.Total_Claim_Amount.mode()

ewa.dtypes

# Third moment business decision

ewa.Customer_Lifetime_Value.skew()
ewa.Income.skew()
ewa.Monthly_Premium_Auto.skew()
ewa.Months_Since_Last_Claim.skew()
ewa.Months_Since_Policy_Inception.skew()
ewa.Number_of_Open_Complaints.skew()
ewa.Number_of_Policies.skew()
ewa.Total_Claim_Amount.skew()

# Fourth moment business decision

ewa.Customer_Lifetime_Value.kurt()
ewa.Income.kurt()
ewa.Monthly_Premium_Auto.kurt()
ewa.Months_Since_Last_Claim.kurt()
ewa.Months_Since_Policy_Inception.kurt()
ewa.Number_of_Open_Complaints.kurt()
ewa.Number_of_Policies.kurt()
ewa.Total_Claim_Amount.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

plt.hist(ewa.Customer_Lifetime_Value) #histogram
plt.hist(ewa.Income, color='red')
plt.hist(ewa.Monthly_Premium_Auto, color='black')
plt.hist(ewa.Months_Since_Last_Claim, color='blue')
plt.hist(ewa.Months_Since_Policy_Inception, color='yellow')
plt.hist(ewa.Number_of_Open_Complaints, color='orange')
plt.hist(ewa.Number_of_Policies, color='pink')
plt.hist(ewa.Total_Claim_Amount, color='indigo')

plt.boxplot(ewa.Customer_Lifetime_Value) #boxplot
plt.boxplot(ewa.Income) #boxplot
plt.boxplot(ewa.Monthly_Premium_Auto) #boxplot
plt.boxplot(ewa.Months_Since_Last_Claim) #boxplot
plt.boxplot(ewa.Months_Since_Policy_Inception) #boxplot
plt.boxplot(ewa.Number_of_Open_Complaints) #boxplot
plt.boxplot(ewa.Number_of_Policies) #boxplot
plt.boxplot(ewa.Total_Claim_Amount) #boxplot
_______________________________________________________________________________
# outlier treatment
import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
ewa.dtypes

# let's find outliers 

sns.boxplot(ewa.Customer_Lifetime_Value)
sns.boxplot(ewa.Income) # no outliers
sns.boxplot(ewa.Monthly_Premium_Auto)
sns.boxplot(ewa.Months_Since_Last_Claim) # no outliers
sns.boxplot(ewa.Months_Since_Policy_Inception) # no outliers
sns.boxplot(ewa.Number_of_Open_Complaints)
sns.boxplot(ewa.Number_of_Policies)
sns.boxplot(ewa.Total_Claim_Amount)

# So, we have 5 variables which has outliers
________________________________________________________________________________
# 1. Customer_Lifetime_Value
# Detection of outliers 
IQR = ewa['Customer_Lifetime_Value'].quantile(0.75) - ewa['Customer_Lifetime_Value'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Customer_Lifetime_Value'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Customer_Lifetime_Value'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Customer_Lifetime_Value'] > upper_limit, True, np.where(ewa['Customer_Lifetime_Value'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Customer_Lifetime_Value)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Customer_Lifetime_Value'] > upper_limit, upper_limit, np.where(ewa['Customer_Lifetime_Value'] < lower_limit, lower_limit, ewa['Customer_Lifetime_Value'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers
________________________________________________________________________________
# 2. Monthly_Premium_Auto
# Detection of outliers 
IQR = ewa['Monthly_Premium_Auto'].quantile(0.75) - ewa['Monthly_Premium_Auto'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Monthly_Premium_Auto'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Monthly_Premium_Auto'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Monthly_Premium_Auto'] > upper_limit, True, np.where(ewa['Monthly_Premium_Auto'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Monthly_Premium_Auto)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Monthly_Premium_Auto'] > upper_limit, upper_limit, np.where(ewa['Monthly_Premium_Auto'] < lower_limit, lower_limit, ewa['Monthly_Premium_Auto'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers
________________________________________________________________________________
# 3. Number_of_Open_Complaints
# Detection of outliers 
IQR = ewa['Number_of_Open_Complaints'].quantile(0.75) - ewa['Number_of_Open_Complaints'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Number_of_Open_Complaints'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Number_of_Open_Complaints'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Number_of_Open_Complaints'] > upper_limit, True, np.where(ewa['Number_of_Open_Complaints'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Number_of_Open_Complaints)
# we see no outiers
________________________________________________________________________________
# 4. Number_of_Policies
# Detection of outliers 
IQR = ewa['Number_of_Policies'].quantile(0.75) - ewa['Number_of_Policies'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Number_of_Policies'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Number_of_Policies'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Number_of_Policies'] > upper_limit, True, np.where(ewa['Number_of_Policies'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Number_of_Policies)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Number_of_Policies'] > upper_limit, upper_limit, np.where(ewa['Number_of_Policies'] < lower_limit, lower_limit, ewa['Number_of_Policies'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers

________________________________________________________________________________
# 5. Total_Claim_Amount
# Detection of outliers 
IQR = ewa['Total_Claim_Amount'].quantile(0.75) - ewa['Total_Claim_Amount'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Total_Claim_Amount'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Total_Claim_Amount'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Total_Claim_Amount'] > upper_limit, True, np.where(ewa['Total_Claim_Amount'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Total_Claim_Amount)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Total_Claim_Amount'] > upper_limit, upper_limit, np.where(ewa['Total_Claim_Amount'] < lower_limit, lower_limit, ewa['Total_Claim_Amount'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers
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
# Dummy Variables 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
ewa.dtypes

# drop emp_name column
ewa.drop(['Customer'], axis=1, inplace=True)
ewa.drop(['State'], axis=1, inplace=True)
ewa.drop(['Effective_To_Date'], axis=1, inplace=True)
ewa.dtypes

# Create dummy variables
ewa_new = pd.get_dummies(ewa)
ewa_new_1 = pd.get_dummies(ewa, drop_first = True)
# we have created dummies for all categorical columns

##### One Hot Encoding works
ewa.columns

from sklearn.preprocessing import OneHotEncoder
# Creating instance of One Hot Encoder
enc = OneHotEncoder() # initializing method
#######################
# Label Encoder
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

# Data Split into Input and Output variables
X = ewa.iloc[:, [1,2,3,4,5,7,8,14,15,16,17,19,20]]
y = ewa.iloc[:, [0,6,9,10,11,12,13,18]] # Moving columns which are not needed for encoding into y
ewa.dtypes

X['Response']= labelencoder.fit_transform(X['Response'])
X['Coverage'] = labelencoder.fit_transform(X['Coverage'])
X['Education'] = labelencoder.fit_transform(X['Education'])
X['EmploymentStatus']= labelencoder.fit_transform(X['EmploymentStatus'])
X['Gender'] = labelencoder.fit_transform(X['Gender'])
X['Location_Code'] = labelencoder.fit_transform(X['Location_Code'])
X['Marital_Status']= labelencoder.fit_transform(X['Marital_Status'])
X['Policy_Type'] = labelencoder.fit_transform(X['Policy_Type'])
X['Policy'] = labelencoder.fit_transform(X['Policy'])
X['Renew_Offer_Type'] = labelencoder.fit_transform(X['Renew_Offer_Type'])
X['Sales_Channel'] = labelencoder.fit_transform(X['Sales_Channel'])
X['Vehicle_Class'] = labelencoder.fit_transform(X['Vehicle_Class'])
X['Vehicle_Size']= labelencoder.fit_transform(X['Vehicle_Size'])

### label encode y ###
y = pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function
# concatenate X and y
ewa_new = pd.concat([X, y], axis =1)

## rename column name
ewa_new.columns
ewa_new.isna().sum()

_______________________________________________________________________________
## Standardization and Normalization #########
import pandas as pd
import numpy as np

### Standardization
from sklearn.preprocessing import StandardScaler
ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
ewa.dtypes

# drop emp_name column
ewa.drop(["Customer","Vehicle_Size","Vehicle_Class","Sales_Channel","Renew_Offer_Type","Policy","Policy_Type","Marital_Status","Gender","Location_Code","State","Effective_To_Date","Response","Coverage","Education","EmploymentStatus"], axis=1, inplace=True)

a = ewa.describe()
# Initialise the Scaler
scaler = StandardScaler()
# To scale data
df = scaler.fit_transform(ewa)
# Convert the array back to a dataframe
dataset = pd.DataFrame(df)
res = dataset.describe()


### Normalization
## load data set
ewa = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
ewa.dtypes
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
import pandas as pd
import matplotlib.pylab as plt
auto = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
auto.describe()
auto.info()
autonew = auto.drop(["Customer","State","Effective_To_Date"], axis=1)
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
# Dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")
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
df_norm['clust'] = cluster_labels # creating a new column and assigning it to new column 
autonew = df_norm.iloc[:, [48,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,30.41,42,43,44,45,46,47]]
auto.head()
# inferences
# count the number in each cluster
auto['clust'].value_counts()
# Aggregate mean of each cluster
auto.iloc[:, 0:].groupby(auto.clust).mean()
autonew.iloc[:, 0:].groupby(autonew.clust).mean()






import pandas as pd
import numpy as np
import matplotlib.pylab as plt

ewa1 = pd.read_csv(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\AutoInsurance.csv")
ewa1.dtypes
ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["Customer","State","Response","Coverage","Education","Effective_To_Date","EmploymentStatus","Gender","Location_Code","Marital_Status","Policy_Type","Policy","Renew_Offer_Type","Sales_Channel","Vehicle_Class","Vehicle_Size"], axis=1)

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

ewa = ewa1.iloc[:, [9,1,2,3,4,5,6,7,8]]
ewa1.head()

# Aggregate mean of each cluster
ewa1.iloc[:, 1:].groupby(ewa1.clust).mean()
ewa1['clust'].value_counts()
# creating a csv file 
ewa1.to_csv("Telco_customer_churn.csv", encoding = "utf-8")


import os
os.getcwd()