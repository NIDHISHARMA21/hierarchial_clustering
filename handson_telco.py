import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
ewa.dtypes
# Exploratory Data Analysis
ewa.describe()
# Measures of Central Tendency / First moment business decision
ewa.Number_of_Referrals.mean()
ewa.Number_of_Referrals.median()
ewa.Number_of_Referrals.mode()

ewa.Tenure_in_Months.mean()
ewa.Tenure_in_Months.median()
ewa.Tenure_in_Months.mode()

ewa.Avg_Monthly_Long_Distance_Charges.mean()
ewa.Avg_Monthly_Long_Distance_Charges.median()
ewa.Avg_Monthly_Long_Distance_Charges.mode()

ewa.Avg_Monthly_GB_Download .mean()
ewa.Avg_Monthly_GB_Download .median()
ewa.Avg_Monthly_GB_Download .mode()

ewa.Monthly_Charge.mean() 
ewa.Monthly_Charge.median()
ewa.Monthly_Charge.mode()


ewa.Total_Charges.mean() 
ewa.Total_Charges.median()
ewa.Total_Charges.mode()

ewa.Total_Refunds.mean()
ewa.Total_Refunds.median()
ewa.Total_Refunds.mode()

ewa.Total_Extra_Data_Charges.mean()
ewa.Total_Extra_Data_Charges.median()
ewa.Total_Extra_Data_Charges.mode()

ewa.Total_Long_Distance_Charges.mean() 
ewa.Total_Long_Distance_Charges.median()
ewa.Total_Long_Distance_Charges.mode()

ewa.Total_Revenue.mean() 
ewa.Total_Revenue.median()
ewa.Total_Revenue.mode()
ewa.dtypes

# Third moment business decision

ewa.Number_of_Referrals.skew()
ewa.Tenure_in_Months.skew()
ewa.Avg_Monthly_Long_Distance_Charges.skew()
ewa.Avg_Monthly_GB_Download.skew()
ewa.Monthly_Charge.skew()
ewa.Total_Charges.skew()
ewa.Total_Refunds.skew()
ewa.Total_Extra_Data_Charges.skew()
ewa.Total_Long_Distance_Charges.skew()
ewa.Total_Revenue.skew()

# Fourth moment business decision

ewa.Number_of_Referrals.kurt()
ewa.Tenure_in_Months.kurt()
ewa.Avg_Monthly_Long_Distance_Charges.kurt()
ewa.Avg_Monthly_GB_Download.kurt()
ewa.Monthly_Charge.kurt()
ewa.Total_Charges.kurt()
ewa.Total_Refunds.kurt()
ewa.Total_Extra_Data_Charges.kurt()
ewa.Total_Long_Distance_Charges.kurt()
ewa.Total_Revenue.kurt()

# Data Visualization
import matplotlib.pyplot as plt
import numpy as np

plt.hist(ewa.Count) #histogram
plt.hist(ewa.Number_of_Referrals, color='red')
plt.hist(ewa.Tenure_in_Months, color='black')
plt.hist(ewa.Avg_Monthly_Long_Distance_Charges, color='blue')
plt.hist(ewa.Avg_Monthly_GB_Download, color='yellow')
plt.hist(ewa.Monthly_Charge, color='orange')
plt.hist(ewa.Total_Charges, color='pink')
plt.hist(ewa.Total_Refunds, color='indigo')
plt.hist(ewa.Total_Extra_Data_Charges, color='violet')
plt.hist(ewa.Total_Long_Distance_Charges, color='green')
plt.hist(ewa.Total_Revenue, color='red')


plt.boxplot(ewa.Number_of_Referrals) #boxplot
plt.boxplot(ewa.Tenure_in_Months) #boxplot
plt.boxplot(ewa.Avg_Monthly_Long_Distance_Charges) #boxplot
plt.boxplot(ewa.Avg_Monthly_GB_Download) #boxplot
plt.boxplot(ewa.Monthly_Charge) #boxplot
plt.boxplot(ewa.Total_Charges) #boxplot
plt.boxplot(ewa.Total_Refunds) #boxplot
plt.boxplot(ewa.Total_Extra_Data_Charges) #boxplot
plt.boxplot(ewa.Total_Long_Distance_Charges) #boxplot
plt.boxplot(ewa.Total_Revenue) #boxplot

_______________________________________________________________________________
# outlier treatment
import pandas as pd
import numpy as np
import seaborn as sns

ewa = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
ewa.dtypes

# let's find outliers 

sns.boxplot(ewa.Number_of_Referrals)
sns.boxplot(ewa.Tenure_in_Months) # no outliers
sns.boxplot(ewa.Avg_Monthly_Long_Distance_Charges) # no outliers
sns.boxplot(ewa.Avg_Monthly_GB_Download)
sns.boxplot(ewa.Monthly_Charge) # no outliers
sns.boxplot(ewa.Total_Charges) # no outliers
sns.boxplot(ewa.Total_Refunds)
sns.boxplot(ewa.Total_Extra_Data_Charges)
sns.boxplot(ewa.Total_Long_Distance_Charges)
sns.boxplot(ewa.Total_Revenue)

# So, we have 6 variables which has outliers
________________________________________________________________________________
# 1. Number_of_Referrals
# Detection of outliers 
IQR = ewa['Number_of_Referrals'].quantile(0.75) - ewa['Number_of_Referrals'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Number_of_Referrals'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Number_of_Referrals'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Number_of_Referrals'] > upper_limit, True, np.where(ewa['Number_of_Referrals'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Number_of_Referrals)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Number_of_Referrals'] > upper_limit, upper_limit, np.where(ewa['Number_of_Referrals'] < lower_limit, lower_limit, ewa['Number_of_Referrals'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers
________________________________________________________________________________
# 2. Avg_Monthly_GB_Download
# Detection of outliers 
IQR = ewa['Avg_Monthly_GB_Download'].quantile(0.75) - ewa['Avg_Monthly_GB_Download'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Avg_Monthly_GB_Download'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Avg_Monthly_GB_Download'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Avg_Monthly_GB_Download'] > upper_limit, True, np.where(ewa['Avg_Monthly_GB_Download'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Avg_Monthly_GB_Download)
# we see no outiers
________________________________________________________________________________
# 3. Total_Refunds
# Detection of outliers 
IQR = ewa['Total_Refunds'].quantile(0.75) - ewa['Total_Refunds'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Total_Refunds'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Total_Refunds'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Total_Refunds'] > upper_limit, True, np.where(ewa['Total_Refunds'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Total_Refunds)
# we see no outiers
________________________________________________________________________________
# 4. Total_Extra_Data_Charges
# Detection of outliers 
IQR = ewa['Total_Extra_Data_Charges'].quantile(0.75) - ewa['Total_Extra_Data_Charges'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Total_Extra_Data_Charges'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Total_Extra_Data_Charges'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Total_Extra_Data_Charges'] > upper_limit, True, np.where(ewa['Total_Extra_Data_Charges'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Total_Extra_Data_Charges)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Total_Extra_Data_Charges'] > upper_limit, upper_limit, np.where(ewa['Total_Extra_Data_Charges'] < lower_limit, lower_limit, ewa['Total_Extra_Data_Charges'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers

________________________________________________________________________________
# 5. Total_Long_Distance_Charges
# Detection of outliers 
IQR = ewa['Total_Long_Distance_Charges'].quantile(0.75) - ewa['Total_Long_Distance_Charges'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Total_Long_Distance_Charges'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Total_Long_Distance_Charges'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Total_Long_Distance_Charges'] > upper_limit, True, np.where(ewa['Total_Long_Distance_Charges'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Total_Long_Distance_Charges)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Total_Long_Distance_Charges'] > upper_limit, upper_limit, np.where(ewa['Total_Long_Distance_Charges'] < lower_limit, lower_limit, ewa['Total_Long_Distance_Charges'])))
sns.boxplot(ewa.ewa_replaced)# we see no outiers
________________________________________________________________________________
# 6. Total_Revenue
# Detection of outliers 
IQR = ewa['Total_Revenue'].quantile(0.75) - ewa['Total_Revenue'].quantile(0.25) # IQR - Inter quartile range IQR = Q3-Q1
lower_limit = ewa['Total_Revenue'].quantile(0.25) - (IQR * 1.5) # Q1 - 1.5 * IQR
upper_limit = ewa['Total_Revenue'].quantile(0.75) + (IQR * 1.5) # Q3 + 1.5 * IQR

############### 1. Remove (let's trim the dataset) ################

# Trimming Technique
# let's flag the outliers in the data set

outliers_ewa = np.where(ewa['Total_Revenue'] > upper_limit, True, np.where(ewa['Total_Revenue'] < lower_limit, True, False))
ewa_trimmed = ewa.loc[~(outliers_ewa), ]
ewa.shape, ewa_trimmed.shape
sns.boxplot(ewa_trimmed.Total_Revenue)
# we see outiers

############### 2.Replace ###############
# Now let's replace the outliers by the maximum and minimum limit

ewa['ewa_replaced'] = pd.DataFrame(np.where(ewa['Total_Revenue'] > upper_limit, upper_limit, np.where(ewa['Total_Revenue'] < lower_limit, lower_limit, ewa['Total_Revenue'])))
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
ewa.drop(['Count'], axis=1, inplace=True)
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

ewa = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
ewa.dtypes

# drop emp_name column
ewa.drop(['Customer_ID'], axis=1, inplace=True)
ewa.drop(['Count'], axis=1, inplace=True)
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
X = ewa.iloc[:, [0,1,4,5,7,8,9,11,12,13,14,15,16,17,18,19,20,21]]
y = ewa.iloc[:, [2,3,6,10,22,23,24,25,26,27]] # Moving columns which are not needed for encoding into y
ewa.dtypes

X['Quarter']= labelencoder.fit_transform(X['Quarter'])
X['Referred_a_Friend'] = labelencoder.fit_transform(X['Referred_a_Friend'])
X['Offer'] = labelencoder.fit_transform(X['Offer'])
X['Phone_Service']= labelencoder.fit_transform(X['Phone_Service'])
X['Multiple_Lines'] = labelencoder.fit_transform(X['Multiple_Lines'])
X['Internet_Service'] = labelencoder.fit_transform(X['Internet_Service'])
X['Internet_Type']= labelencoder.fit_transform(X['Internet_Type'])
X['Online_Security'] = labelencoder.fit_transform(X['Online_Security'])
X['Online_Backup'] = labelencoder.fit_transform(X['Online_Backup'])
X['Device_Protection_Plan'] = labelencoder.fit_transform(X['Device_Protection_Plan'])
X['Streaming_Movies'] = labelencoder.fit_transform(X['Streaming_Movies'])
X['Streaming_Music'] = labelencoder.fit_transform(X['Streaming_Music'])
X['Unlimited_Data']= labelencoder.fit_transform(X['Unlimited_Data'])
X['Contract'] = labelencoder.fit_transform(X['Contract'])
X['Paperless_Billing'] = labelencoder.fit_transform(X['Paperless_Billing'])
X['Payment_Method']= labelencoder.fit_transform(X['Payment_Method'])
X['Premium_Tech_Support']= labelencoder.fit_transform(X['Premium_Tech_Support'])
X['Streaming_TV']= labelencoder.fit_transform(X['Streaming_TV'])

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
ewa = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
ewa.dtypes

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
ewa = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
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
ewa1 = pd.read_excel(r"C:\6. hierarchial_clustering\Dataset_Assignment Clustering\Telco_customer_churn.xlsx")
ewa1.describe()
ewa1.info()

ewa = ewa1.drop(["Customer_ID","Count","Quarter","Referred_a_Friend","Offer","Phone_Service","Multiple_Lines","Internet_Service","Internet_Type","Online_Security","Online_Backup","Device_Protection_Plan","Streaming_Movies","Streaming_Music","Unlimited_Data","Contract","Paperless_Billing","Payment_Method","Premium_Tech_Support","Streaming_TV"], axis=1)

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
ewa1 = ewa.iloc[:, [10,1,2,3,4,5,6,7,8,9]]
ewa1.head()
# Aggregate mean of each cluster
ewa1.iloc[:, 0:].groupby(ewa1.clust).mean()
ewa1['clust'].value_counts()
# creating a csv file 
ewa1.to_csv("Telco_customer_churn.csv", encoding = "utf-8")
