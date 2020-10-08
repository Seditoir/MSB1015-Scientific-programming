### scientific programming heart failure dataset
# Author: Tom Schoenmakers
###

## Data and module loading
# input: Tom_data.xlsx(external data file)
# output: df


## Data and module loading
# input: Tom_data.xlsx(external data file)
# output: df


# Importing neccesary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  #for this document to work needs to be seaborn 0.11 or higher

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import plotly
import plotly.graph_objs as go

from sklearn.ensemble import IsolationForest


# Loading in the heart failure data
df = pd.read_excel (r'Tom_data.xlsx')


# Preliminary data exploration
print(df.dtypes)
print(list(df.columns))
print(df.head())




## changing logical inconsistencies in the dataframe
# input: df
# output: df

# convert the decimal numbers to interger numbers for the age column
df["age"] = df["age"].apply(np.int64)


#   ejection fraction has a negative value (-5) and a impossible high value (3.5e+06), these values are replaced with nan
df = df.replace(-5, np.nan)
df = df.replace(3.5e+06,np.nan)


#   serum_sodium has a impossible high value (2e+09), this is replaced with nan, there is also a 1 which is highly unlikely.
df = df.replace(2e+09,np.nan)
df["serum_sodium"] = df["serum_sodium"].replace(1,np.nan)



## pre-preprocess data exploration
# input: df
# output: df_binary (containing only binary variables, time, and death_event)
#         df_interger_fraction (Containing only interger, fractional, time, and death event)


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


# violin plot per variable, violin plot death per variable, and joint plot for the interger variable type
for column in df_interger_fraction :
    plt.figure()
    sns.violinplot(y=df_interger_fraction[column],width= 0.2)
  
for column in df_interger_fraction :
    plt.figure()
    sns.violinplot(x=df_interger_fraction ['DEATH_EVENT'], y=df_interger_fraction[column])
    
for column in df_interger_fraction :
    plt.figure()
    sns.jointplot(data=df_interger_fraction, x="time", y=column, hue="DEATH_EVENT")

    
# heatmap of the continouse data
plt.figure(figsize=(10,8))
sns.heatmap(df_interger_fraction)


# histogram plots for the binary 
for column in df_binary :
    plt.figure()
    sns.histplot(data=df_binary, x=column)
 
    
 
## data scaling 
# Input: df
# Output: df_scaled(data scaled per variable to a range between 0 and 1)

# scaling of the non binary variable collumns for distance based methods
scaler = MinMaxScaler()
print(scaler.fit(df))

# using the min max scaler and then converting the numpy array back to a pandas array
df_scaled = scaler.transform(df)
df_scaled = pd.DataFrame(data=df_scaled, columns=df.columns.tolist())



# heatmap comparison
fig,(ax1,ax2)= plt.subplots(nrows=2,figsize=(10,16),sharex=True)
sns.heatmap(df,ax=ax2).set_title("unscaled")
sns.heatmap(df_scaled,ax=ax1).set_title("scaled")


# violin plots to confirm scaling
for column in df_scaled :
       fig, (ax1, ax2) = plt.subplots(ncols=2)
       sns.violinplot(y=df_scaled[column],width= 0.2, ax=ax1).set_title("scaled")
       sns.violinplot(y=df[column],width= 0.2, ax=ax2).set_title("unscaled")
        
       
 ## imputation of scaled dataframe
# Input: df_scaled
# Output: df_scaled_imp_median (imputation using the median of a column)
#         df_scaled_imp_iterative (imputation using an iterative process)
#         df_scaled_knn1 (imputation using knn with n=1)
#         df_scaled_knn3 (imputation using knn with n=3)
#         df_scaled_knn5 (imputation using knn with n=5)


# function calculating the amount of misisng values in total, per column, per row, and
#   total amount of rows containg missing values 
# input: pandas dataframe
def imputation_stats(dataframe):
    missing_values_per_column = dataframe.isnull().sum()
    missing_values_total = dataframe.isnull().sum().sum()
    missing_values_per_row = []

    for i in range(len(dataframe.index)) :
        x = (dataframe.iloc[i].isnull().sum())
        missing_values_per_row.append(x)

    missing_values_per_row = np.array(missing_values_per_row)
    missing_values_in_rows = missing_values_per_row.sum()
    percentage_of_missing_values = (missing_values_in_rows/len(dataframe))*100

    # output results of missing values amount before imputation
    print('missing values:',missing_values_total)
    print('missing values per row:',missing_values_in_rows)
    print('missing values per row precentage',percentage_of_missing_values,'%')

    
# knn impuation of the missing values.

imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
imp_iterative = IterativeImputer(max_iter=10, random_state=42)
knn_1 = KNNImputer(n_neighbors=1)
knn_3 = KNNImputer(n_neighbors=3)
knn_5 = KNNImputer(n_neighbors=5)

df_scaled_imp_median = imp_median.fit_transform(df_scaled)
df_scaled_imp_iterative = imp_iterative.fit_transform(df_scaled)
df_scaled_knn1 = knn_1.fit_transform(df_scaled)
df_scaled_knn3 = knn_3.fit_transform(df_scaled)
df_scaled_knn5 = knn_5.fit_transform(df_scaled)

df_scaled_imp_median = pd.DataFrame(data=df_scaled_imp_median, columns=df.columns.tolist())
df_scaled_imp_iterative = pd.DataFrame(data=df_scaled_imp_iterative, columns=df.columns.tolist())
df_scaled_knn1 = pd.DataFrame(data=df_scaled_knn1, columns=df.columns.tolist())
df_scaled_knn3 = pd.DataFrame(data=df_scaled_knn3, columns=df.columns.tolist())
df_scaled_knn5 = pd.DataFrame(data=df_scaled_knn5, columns=df.columns.tolist())

#printing results of imputation
print('-----before imputation')
imputation_stats(df_scaled)
print('-----single variate imputation based on median')
imputation_stats(df_scaled_imp_median)
print('-----irative imputation')
imputation_stats(df_scaled_imp_iterative)
print('-----KNN n=1 imputation')
imputation_stats(df_scaled_knn1)
print('-----KNN n=3 imputation')
imputation_stats(df_scaled_knn3)
print('-----KNN n=5 imputation')
imputation_stats(df_scaled_knn5)


## visualization of imputation
# input: df_scaled_imp_median
#         df_scaled_imp_iterative
#         df_scaled_knn1
#         df_scaled_knn3
#         df_scaled_knn5
# output: df_scaled_imputed (set using results obtained)

# visiulization using violin plots
for column in df_scaled :
       fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(ncols=6,figsize=(16,8))
       sns.violinplot(y=df_scaled[column],width= 0.2, ax=ax1).set_title("Before imputation")
       sns.violinplot(y=df_scaled_imp_median[column],width= 0.2, ax=ax2).set_title("median imputation")
       sns.violinplot(y=df_scaled_imp_iterative[column],width= 0.2, ax=ax3).set_title("irative imputation")
       sns.violinplot(y=df_scaled_knn1[column],width= 0.2, ax=ax4).set_title("knn n=1")
       sns.violinplot(y=df_scaled_knn3[column],width= 0.2, ax=ax5).set_title("knn n=3")
       sns.violinplot(y=df_scaled_knn5[column],width= 0.2, ax=ax6).set_title("knn n=5")
        
#heatmaps
fig,(ax1, ax2, ax3, ax4, ax5, ax6)= plt.subplots(nrows=6,figsize=(10,48),sharex=True)
sns.heatmap(df_scaled,ax=ax1).set_title("before imputation")
sns.heatmap(df_scaled_imp_median,ax=ax2).set_title("median imputation")
sns.heatmap(df_scaled_imp_iterative,ax=ax3).set_title("irative imputation")
sns.heatmap(df_scaled_knn1,ax=ax4).set_title("knn n=1")
sns.heatmap(df_scaled_knn3,ax=ax5).set_title("knn n=3")
sns.heatmap(df_scaled_knn5,ax=ax6).set_title("knn5 n=5")

# based on the results an imputation method is set here 
df_scaled_imputed = df_scaled_knn3

       


## PCA 

# note to self only do the pca on the contineuos data
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# subset continous data for pca

df_pca = df_scaled_imputed[['age',
                           'creatinine_phosphokinase',
                           'ejection_fraction',
                           'platelets',
                           'serum_creatinine',
                           'serum_sodium',
                           'time']]


# PCA function
def pca_function(continous_data,whole_data):
    pca = PCA(n_components=7,svd_solver='full')

    principalComponents = pca.fit_transform(continous_data)
    principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 
                          'principal component 2',
                          'principal component 3',
                          'principal component 4',
                          'principal component 5',
                          'principal component 6',
                          'principal component 7'])
                          

    print(pca.explained_variance_ratio_)
    print(sum(pca.explained_variance_ratio_))

    df_PCA_fitted = pd.concat([principalDf, whole_data], axis = 1)
    
    #variance plot
    import scikitplot as skplt
    skplt.decomposition.plot_pca_component_variance(pca)
    plt.show()



    #scree plot
    Scree_Points = pd.DataFrame({'var':pca.explained_variance_ratio_,
                 'PC':['Pc1',
                       'Pc 2',
                       'Pc 3',
                       'Pc 4',
                       'Pc 5',
                       'Pc 6',
                       'Pc 7']})

    plt.figure()
    sns.barplot(data=Scree_Points, x='PC',y="var", color="c");
    plt.xticks(rotation=45)
    plt.tight_layout()

    
    #visualization for PC1,PC2, and PC4
    for column in whole_data :
        plt.figure()
        sns.scatterplot(data=df_PCA_fitted,x="principal component 1", y="principal component 2", hue=column)
    for column in whole_data :
        plt.figure()
        sns.scatterplot(data=df_PCA_fitted,x="principal component 1", y="principal component 3", hue=column)
    for column in whole_data :
        plt.figure()
        sns.scatterplot(data=df_PCA_fitted,x="principal component 1", y="principal component 4", hue=column)
    plt.show()
    
    # heatmap for variance explained by continouis variable
    ax = sns.heatmap(pca.components_,
                 cmap='YlGnBu',
                 yticklabels=[ "PCA"+str(x) for x in range(1,pca.n_components_+1)],
                 xticklabels=list(continous_data.columns))
    plt.show()
    
    

    
# 3d plot 
def plot_3d(pc1,pc2,pc3):
    # Configure Plotly to be rendered inline in the notebook.
    plotly.offline.init_notebook_mode()

    # Configure the trace.
    trace = go.Scatter3d(
        x=pc1,  # <-- Put your data instead
        y=pc2,  # <-- Put your data instead
        z=pc3,  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 2,
            'opacity': 0.8,
        }
    )

    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
           
    )

    data = [trace]

    plot_figure = go.Figure(data=data, layout=layout)

    # Render the plot.
    plotly.offline.iplot(plot_figure)


    
    
    
pca_function(df_pca,df_scaled_imputed)
# plot_3d(df_PCA_fitted["principal component 1"],df_PCA_fitted["principal component 2"],df_PCA_fitted["principal component 3"])








## outlier detection using isolation forest
# input: df_scaled_imputed
# output: df_scaled_imputed_wo_outliers
#         anomaly
#         anomaly_index
#         df_scaled_imputed_anomaly

# outlier detection using isolation forest 
outliers_fraction = 0.05

#prep anomoly detection
df_scaled_imputed_anomaly = df_scaled_imputed

URF=IsolationForest(n_estimators=100, max_samples='auto',random_state=42,contamination=outliers_fraction)
URF.fit(df_scaled_imputed)

#df_scaled_imputed['scores']=URF.decision_function(df_scaled_imputed)
df_scaled_imputed_anomaly['anomaly']=URF.predict(df_scaled_imputed)
df_scaled_imputed_anomaly.head(20)

anomaly = df_scaled_imputed_anomaly.loc[df_scaled_imputed['anomaly']==-1]
anomaly_index = list(anomaly.index)

print('total amount of outliers:',len(anomaly_index))
print('index location:',anomaly_index)


# heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_scaled_imputed)




# dropping the 'Outliers' and rescaling
df_scaled_imputed_wo_outliers  = df_scaled_imputed
df_scaled_imputed_wo_outliers  = df_scaled_imputed_wo_outliers.drop(anomaly_index)
df_scaled_imputed_wo_outliers = df_scaled_imputed_wo_outliers.drop(['anomaly'], axis=1) #dropping column anomaly


scaler = MinMaxScaler()
print(scaler.fit(df_scaled_imputed_wo_outliers))


df_scaled_imputed_wo_outliers_temp = scaler.transform(df_scaled_imputed_wo_outliers)
df_scaled_imputed_wo_outliers = pd.DataFrame(data=df_scaled_imputed_wo_outliers_temp, columns=df.columns.tolist())

print('total samples after removal of outliers:',len(df_scaled_imputed_wo_outliers))


# visualization of outliers dropping
for column in df_scaled_imputed_wo_outliers :
       fig, (ax1, ax2) = plt.subplots(ncols=2)
       sns.violinplot(y=df_scaled_imputed_wo_outliers[column],width= 0.2, ax=ax1).set_title("without outliers")
       sns.violinplot(y=df_scaled_imputed[column],width= 0.2, ax=ax2).set_title("with outliers")
# heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_scaled_imputed_wo_outliers)

#histplot for death events
plt.figure()
sns.histplot(data=df_scaled_imputed, x='DEATH_EVENT').set_title("With outliers")
plt.figure()
sns.histplot(data=df_scaled_imputed_wo_outliers, x='DEATH_EVENT').set_title("without outliers")

#death events statistics

total_death_event_w_outliers = df_scaled_imputed['DEATH_EVENT'].sum()
total_death_event_wo_outliers = df_scaled_imputed_wo_outliers['DEATH_EVENT'].sum()       
total_death_event_w_outliers_percentage =(total_death_event_w_outliers/len(df_scaled_imputed))*100
total_death_event_wo_outliers_percentage =(total_death_event_wo_outliers/len(df_scaled_imputed_wo_outliers))*100

print('total death events with outliers:',total_death_event_w_outliers)
print('total death event without outliers:',total_death_event_wo_outliers)
print('percentage death event with outliers:',total_death_event_w_outliers_percentage, '%')
print('percentage death event without outliers:',total_death_event_wo_outliers_percentage,'%')


# subset continous data for pca

df_pca_2 = df_scaled_imputed_wo_outliers[['age',
                           'creatinine_phosphokinase',
                           'ejection_fraction',
                           'platelets',
                           'serum_creatinine',
                           'serum_sodium',
                           'time']]

pca_function(df_pca_2,df_scaled_imputed_wo_outliers)
#plot_3d(df_PCA_fitted["principal component 1"],df_PCA_fitted["principal component 2"],df_PCA_fitted["principal component 3"])





#biplot

def biplot(score,coeff,labels=None):
    #biplot function 
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()

biplot(principalComponents[:,0:2],np.transpose(pca.components_[0:2, :]),list(df_pca_2.columns))
plt.show()



    