### scientific programming heart failure dataset
# Author: Tom Schoenmakers
###

# Importing neccesary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  #for this document to work needs to be seaborn 0.11 or higher

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest

# Loading in the heart failure data

df = pd.read_excel (r'Tom_data.xlsx')

# Preliminary data exploration
print(df.dtypes)
print(list(df.columns))
print(df.head())

# changing columns (variables) to a logical data type
# convert the decimal numbers to interger numbers for the age column
df["age"] = df["age"].apply(np.int64)

#   ejection fraction has a negative value (-5) and a impossible high value (3.5e+06), these values are replaced with nan
df = df.replace(-5, np.nan)
df = df.replace(3.5e+06,np.nan)

#   serum_sodium has a impossible high value (2e+09), this is replaced with nan
df = df.replace(2e+09,np.nan)

df = df.drop([149], axis=0) # need to be removed if the issue of the 1 concentration of sodium is resolved


# seperating the different data types into sub-data frames for the visualization

df_binary = df[['anaemia',
                'diabetes',
                'high_blood_pressure',
                'sex',
                'smoking',
                'time',
                'DEATH_EVENT']]
df_interger_fraction = df[['age',
                           'creatinine_phosphokinase',
                           'ejection_fraction',
                           'platelets',
                           'serum_creatinine',
                           'serum_sodium',
                           'DEATH_EVENT',
                           'time']]

# to not get memory warnings about the amount of plots

plt.rcParams.update({'figure.max_open_warning': 0})

# violing plot, and joint plot for the interger variable type

for column in df_interger_fraction :
    plt.figure()
    sns.violinplot(y=df_interger_fraction[column],width= 0.2)
#   plt.savefig(column,'_violin.svg')

    
for column in df_interger_fraction :
    plt.figure()
    sns.violinplot(x=df_interger_fraction ['DEATH_EVENT'], y=df_interger_fraction[column])
    
for column in df_interger_fraction :
    plt.figure()
    sns.jointplot(data=df_interger_fraction, x="time", y=column, hue="DEATH_EVENT")

# histogram plots for the binary 

for column in df_binary :
    plt.figure()
    sns.histplot(data=df_binary, x=column)
 
# scaling of the non binary variable collumns for distance based methods
scaler = MinMaxScaler()
print(scaler.fit(df))

# using the min max scaler and then converting the numpy array back to a pandas array
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(data=df_scaled, columns=df.columns.tolist())

print(df_scaled.head())

# plots to confirm scaling

for column in df_scaled :
       fig, (ax1, ax2) = plt.subplots(ncols=2)
       sns.violinplot(y=df_scaled[column],width= 0.2, ax=ax1).set_title("scaled")
       sns.violinplot(y=df[column],width= 0.2, ax=ax2).set_title("unscaled")


# calculating the amount of misisng values in total, per column, per row, and
#   total amount of rows containg missing values

missing_values_per_column = df_scaled.isnull().sum()
missing_values_total = df_scaled.isnull().sum().sum()
missing_values_per_row = []

for i in range(len(df_scaled.index)) :
    x = (df_scaled.iloc[i].isnull().sum())
    missing_values_per_row.append(x)

missing_values_per_row = np.array(missing_values_per_row)
missing_values_in_rows = missing_values_per_row.sum()
percentage_of_missing_values = (missing_values_in_rows/len(df_scaled))*100


# printing results of missing values amount
print('missing values total before imputment:',missing_values_total)
print('missing values per row before imputment:',missing_values_in_rows)
print('missing values per row precentage before imputment:',percentage_of_missing_values,'%')

# svd impuation of the missing values.
imputer = KNNImputer(n_neighbors=5)
df_scaled_imputed = imputer.fit_transform(df_scaled)
df_scaled_imputed = pd.DataFrame(data=df_scaled_imputed, columns=df.columns.tolist())

# repeat form above but for the imputed data

missing_values_per_column = df_scaled_imputed.isnull().sum()
missing_values_total = df_scaled_imputed.isnull().sum().sum()
missing_values_per_row = []

for i in range(len(df_scaled_imputed.index)) :
    x = (df_scaled_imputed.iloc[i].isnull().sum())
    missing_values_per_row.append(x)

missing_values_per_row = np.array(missing_values_per_row)
missing_values_in_rows = missing_values_per_row.sum()
percentage_of_missing_values = (missing_values_in_rows/len(df_scaled))*100

print('missing values total after imputment:',missing_values_total)
print('missing values per row after imputment:',missing_values_in_rows)
print('missing values per row precentage after imputment:',percentage_of_missing_values,'%')


# outlier detection using isolation forest 

URF=IsolationForest(n_estimators=100, max_samples='auto')





















