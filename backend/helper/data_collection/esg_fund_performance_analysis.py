import requests
from bs4 import BeautifulSoup
import re
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import csv
import os
import pickle
from IPython.display import display, HTML
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import graphviz
from sklearn.neighbors import KNeighborsRegressor

# !pip install keras==2.12.0
# !pip uninstall tensorflow
# !pip install tensorflow==2.12.0

import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor # There is also a KerasClassifier class
from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

!pip install openpyxl
import openpyxl
print("Library versions: pandas", pd.__version__," numpy", np.__version__," seaborn", sns.__version__)

def download_data():
    URL = 'https://fossilfreefunds.org/how-it-works'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')

    urls = []
    names = []
    for i, link in enumerate(soup.findAll('a')):
        FULLURL = link.get('href')
        if bool(re.search('.*results.*.xlsx', FULLURL)):
            urls.append(FULLURL)
            names.append(os.path.basename(soup.select('a')[i].attrs['href']))

    names_urls = zip(names, urls)
    for name, url in names_urls:
        print("Download file: "+name)
        r = requests.get(url, verify=False,stream=True)
        r.raw.decode_content = True
        with open("/kaggle/working/" + name, 'wb') as out:
                shutil.copyfileobj(r.raw, out)

def merge_excel():
    df = pd.DataFrame()
    files=os.listdir('data')
    files_xls = [f for f in files if f[-4:]=='xlsx']

    for f in files_xls:
        if not re.match(r".*20210[5-9]+.*", f):
            print('Merging file: '+f)
            data = pd.read_excel('data/'+f, 'Shareclasses',engine='openpyxl')
            df = df.append(data)
    df.to_csv('/kaggle/working/fossilfund_dataset.csv', index=False)
    print('Export to data/fossilfund_dataset.csv is finished')

df = pd.read_csv('/content/fossilfund_dataset.csv')

"""* We can already convert date columns as date type instead of generic object"""

date_cols=df.filter(regex=" date.*",axis=1).columns
df[date_cols]=df[date_cols].apply(pd.to_datetime, errors='coerce')
display(HTML(df[0:10].to_html()))

origin_categories=["Fund profile","Fossil Free Funds", "Deforestation Free Funds", "Gender Equality Funds", "Gun Free Funds", "Prison Free Funds", "Weapon Free Funds", "Tobacco Free Funds", "Financial performance"]
for category in origin_categories:
    print("Distribution of numerical features for category: "+category)
    display(HTML(df.filter(regex=category+".*",axis=1).describe().to_html()))
    print("\n\n")


columns_stats=pd.DataFrame()
columns_stats['fill_percent']=df.notnull().sum(axis=0)/len(df)*100
fig = plt.figure(figsize=(10, 22))
columns_stats['fill_percent'].sort_values().plot.barh()

columns_stats[columns_stats['fill_percent']<60]

df.drop(columns=columns_stats[columns_stats['fill_percent']<60].index.values.tolist(), axis=1, inplace=True)



columns_stats['fill_percent'].filter(regex="Financial performance.*")

"""The "Financial performance: Month end trailing returns, year 1" is the most complete variable as a high percentage of funds have an inception date greater than 2 years (so performance for "year 1" is available) but less than 10 years (so no performance data available for "year 10")"""

perf_date=pd.DataFrame()
for year in [1,3,5,10]:
    filter_df=df['Fund profile: Shareclass inception date'][( (df['Financial performance: Financial performance as-of date'] - pd.DateOffset(years=year)) > df['Fund profile: Shareclass inception date']  )]
    perf_date["year"+str(year)] =filter_df.groupby(filter_df.dt.year).count()

#Normalize row perf_data by year
perf_date_norm=perf_date.div(perf_date.sum(axis=1), axis=0)*100
perf_date_norm.loc['2000-1-1 00:00:00':'2022-1-1 00:00:00'].plot.bar(figsize=(16, 8), stacked=True)

columns_stats=pd.DataFrame()
columns_stats['unique_values']=df.nunique()/len(df)*100
fig = plt.figure(figsize=(10, 22))
columns_stats['unique_values'].sort_values().plot.barh()


cols_df=pd.read_excel('/content/replace.xlsx', 'Cols',  engine='openpyxl')
df.rename(columns=dict(zip(cols_df["Column original"],cols_df["Short column name"])), inplace=True)

def getColCategory(category):
    return list(set(cols_df[cols_df['Category']==category]['Short column name']) & set(df.columns))

def getColType(type_col):
    return list(set(cols_df[cols_df['Type']==type_col]['Short column name']) & set(df.columns))

def getEncoding(shortName):
    return cols_df[cols_df['Short column name']==shortName]['encoding'].values[0]

continuous= getColType('Continuous')
discrete= getColType('Discrete')
ordinal= getColType('Ordinal')
nominal= getColType('Nominal')

date_cols=df.filter(regex=".*Date.*",axis=1).columns


threshold_0_level=10

def zeros_columns(df, col_category):
    zeros_percentage=(df[col_category]==0).sum()*100/len(df[col_category])
    zeros=zeros_percentage[(zeros_percentage>threshold_0_level)].index.values.tolist()
    print("Columns with 0-values > "+str(threshold_0_level)+"% : "+str(len(zeros))+"/"+str(len(col_category)))
    print(zeros_percentage[(zeros_percentage>threshold_0_level)].sort_values(ascending=False))
    return zeros
continuous_zeros = zeros_columns(df, continuous)
discrete_zeros = zeros_columns(df, discrete)


threshold_null_level=2

def null_columns(df, col_category):
    null_percentage=(df[col_category].isnull()).sum()*100/len(df[col_category])
    nulls=null_percentage[(null_percentage>threshold_null_level)].index.values.tolist()
    print("Columns with null-values > "+str(threshold_null_level)+"% : "+str(len(nulls))+"/"+str(len(col_category)))
    print(null_percentage[(null_percentage>threshold_null_level)].sort_values(ascending=False))
    return nulls

discrete_null = null_columns(df, discrete)
continuous_null = null_columns(df, continuous)
nominal_null = null_columns(df, nominal)
ordinal_null = null_columns(df, ordinal)

df.duplicated().sum()


df[nominal].nunique()/len(df)*100


index_col=['FI_ShareclassName', 'FP_PerformanceAs-OfDate']
duplicate_rows=df[df.duplicated(subset=index_col, keep=False)]
duplicate_rows.to_csv('/content/temp.csv', sep=';')
duplicate_rows

origin_len=len(df)
duplicate_len=len(df[df.duplicated(subset=index_col, keep='last')])
df=df[~df.duplicated(subset=index_col, keep='last')]

print("Remaining rows:",len(df),"(",origin_len,"-",duplicate_len,")")
df[df.duplicated(subset=index_col, keep=False)]

df_tmp=df.copy()
df_tmp['FI_AssetManagerFirstLetter']=df_tmp['FI_AssetManager'].str[0]
grouped_df=df_tmp.groupby(df_tmp['FP_PerformanceAs-OfDate'])
grouped_df['FI_AssetManagerFirstLetter'].value_counts()

grouped_df['FI_AssetManagerFirstLetter'].value_counts().groupby(level=0).apply(
    lambda x: x
).unstack().plot.bar(figsize=(16, 8), stacked=True)


per_share_max_count = df.groupby(['FI_ShareclassName'])['FI_ShareclassName'].value_counts().max()
threshold_max_count=0.6

partial_share_missing=df.copy()
partial_share_missing=partial_share_missing.groupby(['FI_ShareclassName']).filter(lambda x: len(x) <= threshold_max_count * per_share_max_count)
partial_share_missing.groupby(['FI_ShareclassName'])['FI_ShareclassName'].value_counts().sort_values(ascending=False)

origin_len=len(df)
df.drop(partial_share_missing.index, inplace=True)
print("Remaining rows:",len(df),"(",origin_len,"-",len(partial_share_missing),")")


for date_col in df.filter(regex="Date.*",axis=1).columns:
    print('Number of empty dates for columns',date_col,":",len(df[df[date_col].isnull()]))
df[df['FP_PerformanceAs-OfDate'].isnull()]

filter=df[df['FP_PerformanceAs-OfDate'].isnull()]
df.loc[filter.index-1,['FP_PerformanceAs-OfDate','FI_PortfolioHoldingsAs-OfDate']]

df.loc[filter.index,'FP_PerformanceAs-OfDate']=df.loc[filter.index-1,'FP_PerformanceAs-OfDate']
df[df['FP_PerformanceAs-OfDate'].isnull()]

def checkuniquevalues(df, cols):
    #Check unique values
    for col in cols:
        print(col,": Total unique:",len(df[col].sort_values().unique())," - Values:",df[col].sort_values().unique())
checkuniquevalues(df, ordinal+nominal)


df.drop(columns=['FI_Ticker','FI_ShareclassTickers','FI_ShareclassName', 'FI_FundName', 'FI_AssetManager'], axis=1, inplace=True)
nominal.remove('FI_Ticker')
nominal.remove('FI_ShareclassTickers')
nominal.remove('FI_ShareclassName')
nominal.remove('FI_FundName')
nominal.remove('FI_AssetManager')

df.to_csv('/content/fossilfund_dataset_clean.csv', index=False)

categories=cols_df['Category'].unique()

for category in categories:
    #index_cols=list(set(cols_df[cols_df['Category']==category]['Short column name']) & set(df.columns))
    index_cols=getColCategory(category)
    length=len(df[index_cols].select_dtypes(exclude=object).columns)
    print("Histogram for "+category+" features")
    df[index_cols].hist(color='g', bins=50, grid=False, figsize=(length*2,length))
    plt.tight_layout()
    plt.show()

# (C) Preprocessing function
def df_wo_zeros_null(df):
    df = df.copy()

    # Continuous
    # Add additional column for holding 0 values
    # Filter-out zero values
    for c in list(continuous_zeros)+list(discrete_zeros):
        name = c + "_isempty"
        idx= df[c]==0
        df[name] = idx
        #Convert bool col as int
        df[name] = df[name].astype(int)
        df[c] = df[~idx][c]

    # Fill missing values
    for c in list(set(continuous + discrete) & set(df.select_dtypes(np.number).columns)):
        df[c].dropna(inplace=True)

    return df

from sklearn.preprocessing import QuantileTransformer

#continuous+discrete
cols= list(set(continuous) & set(df.select_dtypes(np.number).columns))
temp_df=df_wo_zeros_null(df)
for category in categories:
    #index_cols=list(set(cols_df[cols_df['Category']==category]['Short column name']) & set(df.columns))
    index_cols=list( set(getColCategory(category)) & set(cols))
    length=len(index_cols)
    for d in [1, 0.5, 2, 3]:
        test_df=temp_df[index_cols].copy()
        for c in test_df.columns:
            name = '{}**{}'.format(c, d)
            test_df[name]=test_df[c]**d
            test_df.drop(c, axis=1, inplace=True)
        test_df.hist(color='g', bins=30, grid=False, figsize=((length*1.2)+10,length+3))

    if(category != 'Gender Equality'):
        test_df=np.log1p(df[index_cols].copy())
        test_df.columns = [str(col) + '_log1p' for col in test_df.columns]
        test_df.hist(color='g', bins=30, grid=False, figsize=((length*1.2)+10,length+3))
    plt.tight_layout()
    plt.show()

from scipy.stats import boxcox
special_distrib=['FI_PercentRated','F_CarbonMarketValueWeight','GE_WeightOfHoldings']
length=len(special_distrib)
test_df=df[special_distrib].copy()
for feature in special_distrib:
    name = 'expm1({})'.format(feature)
    test_df[name]=(1-test_df[feature])**0.5

    #test_df[name]=boxcox(test_df[feature], 0.3)
    test_df.drop(feature, axis=1, inplace=True)
test_df.hist(color='g', bins=30, grid=False, figsize=((length*1.2)+10,length+3))
plt.tight_layout()
plt.show()


cols= list(set(continuous) & set(df.select_dtypes(np.number).columns))
preprocess_df=df_wo_zeros_null(df)
continuous_log1p=[]
continuous_exp05=[]
continuous_exp1_05=[]

for category in categories:
    #index_cols=list(set(cols_df[cols_df['Category']==category]['Short column name']) & set(df.columns))
    index_cols=list( set(getColCategory(category)) & set(cols))
    length=len(index_cols)
    test_df=preprocess_df[index_cols].copy()
    for c in test_df.columns:
        encoding=getEncoding(c)
        if(encoding == "log1p"):
            name = 'log1p({})'.format(c)
            test_df[name]=np.log1p(test_df[c])
            test_df.drop(c, axis=1, inplace=True)
            continuous_log1p.append(c)
        elif (encoding == "^0.5"):
            name = '{}**{}'.format(c, '0.5')
            test_df[name]=test_df[c]**0.5
            test_df.drop(c, axis=1, inplace=True)
            continuous_exp05.append(c)
        elif (encoding == "(1-x)^0.5"):
            name = '(1-{})^0.5'.format(c)
            test_df[name]=(1-test_df[c])**0.5
            test_df.drop(c, axis=1, inplace=True)
            continuous_exp1_05.append(c)

    test_df.hist(color='g', bins=30, grid=False, figsize=((length*1.2)+10,length+3))
    plt.tight_layout()
    plt.show()

outliers_threshold=3

# (C) Preprocessing function
def preprocess_numerical(df):
    df = df.copy()

    # Continuous
    # Add additional column for holding 0 values
    # Filter-out zero values
    for c in list(continuous_zeros)+list(discrete_zeros):
        name = c + "_isempty"
        idx= df[c]==0
        df[name] = idx
        #Convert bool col as int
        df[name] = df[name].astype(int)
        df[c] = df[~idx][c]


    # Apply feature encoding
    df[continuous_log1p]=np.log1p(df[continuous_log1p])
    df[continuous_exp05]=df[continuous_exp05]**0.5
    df[continuous_exp1_05]=(1-df[continuous_exp1_05])**0.5

    for c in continuous:
        z_scores = (df[c] - df[c].mean()) / df[c].std()
        idx = (np.abs(z_scores) > outliers_threshold)
        df[c] = df[~idx][c]

    # Fill missing values
    for c in list(set(continuous + discrete) & set(df.select_dtypes(np.number).columns)):
        df[c].fillna(df[c].median(), inplace=True)

    #Replace dates
    for date_col in date_cols:
        df[date_col]=pd.to_numeric(df[date_col].apply(pd.to_datetime, errors='coerce'))

    return df

preprocess_df=preprocess_numerical(df)

cols= list(set(continuous) & set(df.select_dtypes(np.number).columns))

for category in categories:
    index_cols=list( set(getColCategory(category)) & set(cols))
    length=len(index_cols)
    print("Histogram for "+category+" features")


for c in preprocess_df.columns:
    print (c)

def ordinal_mapping(df, cols, dictionary):
    for col in cols:
        df[col]=df[col].map(dictionary)
    return df


def preprocessing_categorical(df):
    df = df.copy()
    # convert ordinal to int columns
    ordinal_mapping(df, ordinal, {'A':6, 'B':5, 'C':4, 'D':3, 'E':2, 'F':1, np.nan:0})

    # One-hot encoding
    df = pd.get_dummies(df, columns=nominal, dummy_na=False)

    return df

preprocess_df=preprocessing_categorical(preprocess_df)

print("After processing, we have a total of", len(preprocess_df.columns), "features")

def compareAfter_preprocess(df_origin, df_new):
    number_previous_col=0
    for col in df_new.columns:
        if(col in df_origin.columns ):
            if(not col in continuous):
                number_previous_col=number_previous_col+1
                if(len( list(set(df_new[col].unique()) -set(df_origin[col].unique()) )) >0):
                    print(col,"- difference of value mapping :",list(set(df_new[col].unique()) -set(df_origin[col].unique())))
        else:
            print(col,"- new column")

compareAfter_preprocess(df,preprocess_df)

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_correlations(df, n=10):
    au_corr = df.corr().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

correlation_threshold=0.95

for category in ['FI','FP','F','D','GE','G','W','T','P']:
    filter=preprocess_df.filter(regex=category+"_.*",axis=1).select_dtypes(np.number)
    if(len(filter.columns)>0):
        print("Top Absolute Correlations")
        tmp_corr=get_top_correlations(filter, 40)
        print(tmp_corr[tmp_corr >= correlation_threshold].sort_index())

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    corr_results=[]

    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset
                    corr_results.append({
                        'Deleted column': colname,
                        'Correlation column': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })

    return pd.DataFrame(corr_results)

corr_results=correlation(preprocess_df, correlation_threshold)
pd.set_option('display.max_rows', None)
corr_results

preprocess_df.to_csv('/content/fossilfund_dataset_prep.csv', index=False)

fig = plt.figure(figsize=(25, 20))
sns.heatmap(preprocess_df.reindex(sorted(preprocess_df.columns), axis=1).corr(method='pearson'),
            cmap='RdBu_r',
            annot=False,
            linewidth=0.5)

target = cols_df[ (cols_df['Category']=='Financial performance') & (cols_df['Type']=='Continuous') ]['Short column name']
correlations=preprocess_df.corr(method='pearson')[target]#.sort_values(ascending=False)
print('Top positive correlations with ReturnsY1')
correlations['FP_ReturnsY1'].sort_values(ascending=False)[0:15]

print('Top negative correlations with ReturnsY1')
correlations['FP_ReturnsY1'].sort_values(ascending=False)[-5:]

correlations.filter(regex="FI_AssetManager.*",axis=0)['FP_ReturnsY1'].sort_values(ascending=False)
target = 'FP_ReturnsY1'
nb_samples=5000
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

for category in ['FI','FP','F','D','GE','G','W','T','P']:
    index_cols=preprocess_df.filter(regex="^"+category+"_.*",axis=1).columns
    index_cols_numb_only=preprocess_df[index_cols].select_dtypes(exclude=object).columns
    index_cols_list=list(chunks(index_cols_numb_only, 5))
    number_sets=len(index_cols_list)
    for i in range(0,len(index_cols_list)):
        print("Pairplots for "+category+" numeric features - set "+str(i+1)+"/"+str(number_sets))

for date_col in date_cols:
    preprocess_df[date_col]=preprocess_df[date_col].apply(pd.to_datetime, errors='coerce')

preprocess_df['FP_ReturnsY1_diff']=df['FP_ReturnsY1'].diff()

test_size_split=0.4
clean_df=pd.read_csv('/content/fossilfund_dataset_prep.csv')

print("How many non numerical columns:",len(clean_df.select_dtypes(['number']).columns)-len(clean_df.columns))

#Remove FP
clean_df.drop(columns=['FP_ReturnsY3','FP_ReturnsY5','FP_ReturnsY10'], axis=1, inplace=True)

#All rows contain null values
print("No. of rows containing null values", len(clean_df.isna().sum(axis=1).eq(0) ))
print("Total no. of columns in the dataframe", len(clean_df.columns))
print("No. of numerical columns ", len(clean_df.select_dtypes(np.number).columns))
print("No. of columns containing null values", len(clean_df.columns[clean_df.isna().any()]))
print("Details of columns containing null values", clean_df.columns[clean_df.isna().any()])
print("No. of columns not containing null values", len(clean_df.columns[clean_df.notna().all()]))

clean_df[clean_df.isna().any(axis=1)]

"""* Prepare X and y variables for the SelectKBest"""

target = 'FP_ReturnsY1'
X = clean_df.drop(columns=target)
y = clean_df[target]

# How many features do you want to keep?
k = 20

# Create the selecter object
skb = SelectKBest(f_regression, k=k)

# Fit the selecter to your data
X_new = skb.fit_transform(X, y)

# Extract the top k features from the `pvalues_` attribute
k_feat = np.argsort(skb.pvalues_)[:k]

# Reduce the dataframe according to the selecter
df_reduced = clean_df[X.columns[k_feat]]

# instantiate SelectKBest to determine 20 best features
best_features = SelectKBest(score_func=f_regression, k=k)
fit = best_features.fit(X,y)
df_scores = pd.DataFrame(fit.scores_)
df_columns = pd.DataFrame(X.columns)
# concatenate dataframes
feature_scores = pd.concat([df_columns, df_scores],axis=1)
feature_scores.columns = ['Feature_Name','Score']  # name output columns
topk_features=feature_scores.nlargest(k,'Score')
print(topk_features)  # print top k best features


"""
We define our 1 model
- Grade: we keep only grade features, except for the Gender equality category where *GE_WeightOfHoldings* has better scoring than the grading feature and discard *G_CivilianFirearmGrade* feature which has a too low score to be included
"""

features_grade_model=['F_FossilFuelGrade',
'P_PrisonIndustrialComplexGrade',
'D_DeforestationGrade',
'T_TobaccoGrade',
'W_MilitaryWeaponGrade',
'GE_WeightOfHoldings']

def splitTrainTest(df,target,features):
    #split current dataframe into train set / test set
    # Create X, y
    X = df[features]
    y = df[target]
    # Split into train/test sets
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size_split, random_state=0)
    # Standardize data
    scaler = StandardScaler()
    X_tr_rescaled = scaler.fit_transform(X_tr)
    X_te_rescaled = scaler.transform(X_te)

    return X_tr_rescaled, X_te_rescaled, y_tr, y_te, scaler


def saveModelResults(model, modelName, X_te_rescaled, y_te):
    mae = MAE(y_te, model.predict(X_te_rescaled))
    print('MAE with best alpha: {:,.3f}%'.format(mae))

    #Save result
    details = {
        'model' : [modelName],
        'test_accuracy' : [mae],
    }
    df = pd.DataFrame(details)
    df.to_csv('/content/data/results.csv', index=False, mode='a', header=False, float_format='%.3f')
    return [mae, model, modelName]

selected_model=features_grade_model
X_tr_rescaled, X_te_rescaled, y_tr, y_te, scaler = splitTrainTest(clean_df,target,selected_model)

n_neighbors=100
p=2
weights='distance'

neigh = KNeighborsRegressor(n_neighbors=n_neighbors, p=p, weights=weights)
neigh.fit(X_tr_rescaled, y_tr)


kNNResults=saveModelResults(neigh, "KNN", X_te_rescaled, y_te)