### scientific programming heart failure dataset
# Author: Tom Schoenmakers
###

# Importing neccesary packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


 
# Loading in the heart failure data

df = pd.read_excel (r'Tom_data.xlsx')
print(df.dtypes)

# changing columns (variables) to a logical data type
#   ejection fraction has a negative value and a impossible high value
#   serum_sodium has a impossible high value 

df["age"] = df["age"].apply(np.int64)
df = df.drop([149], axis=0)


# boxplot of each column in the data frame

for column in df:
    plt.figure()
    sns.boxplot(y=df[column],width= 0.2)

# df.plot(kind='box')

# violinplot
    
for column in df:
    plt.figure()
    sns.violinplot(x=df['DEATH_EVENT'], y=df[column])   
 



#visualization for each binary column

plt.figure()
sns.distplot(a=df["anaemia"], hist=True, kde=False, rug=False )
plt.show()

plt.figure()
sns.distplot(a=df["diabetes"], hist=True, kde=False, rug=False )
plt.show()

plt.figure()
sns.distplot(a=df["high_blood_pressure"], hist=True, kde=False, rug=False )
plt.show()

plt.figure()
sns.distplot(a=df["sex"], hist=True, kde=False, rug=False )
plt.show()

plt.figure()
sns.distplot(a=df["smoking"], hist=True, kde=False, rug=False )
plt.show()

plt.figure()
sns.distplot(a=df["DEATH_EVENT"], hist=True, kde=False, rug=False )
plt.show()
# URF

# noise removal



# normalization of non binary collumns using probabilistic quotient normalization

# scaling


# calculating the amount of misisng values in total, per column, per row, and
#   total amount of rows containg missing values

missing_values_per_column = df.isnull().sum()
missing_values_total = df.isnull().sum().sum()
missing_values_per_row = []

for i in range(len(df.index)) :
    x = (df.iloc[i].isnull().sum())
    missing_values_per_row.append(x)

missing_values_per_row = np.array(missing_values_per_row)
missing_values_in_rows = missing_values_per_row.sum()
percentage_of_missing_values = (missing_values_in_rows/298)*100

# imputation of missing values using single value decomposition 


3






















