import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
import copy
import pingouin as pg
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error as mse

def handlerOutliers(df, cols):
    pass

def factorize_categorical(df):
    for col in df.select_dtypes(include = "object"):
        df[col] = pd.factorize(df[col])[0]
    return df

def over_50(df):
    for col in df: 
        missing = df[col].isnull().mean()*100 
        if missing > 50: 
           df.drop(col, axis =1,  inplace = True)
        if missing < 5:
            df.dropna(subset = [col], axis = 0, inplace = True)
    return df

def impute_age(df, label, X_test, model):
    nanValues = np.where(df[label].isna())[0]
    y_test = model.predict(X_test).round()
    df.loc[nanValues, label] = y_test
    return df

def process_data(df):
    over_50(df)
    df.drop(["Ticket", "Name", "PassengerId"], axis = 1, inplace = True)
    df = factorize_categorical(df)
    df.reset_index(drop=True, inplace = True)
    return df

def train_model(X_train, Y_train):
    model = LGBMRegressor()
    model.fit(X_train, Y_train)
    return model

def create_splits(df, label):
    train = df[df[label].isna() == 0]
    X_train = train.drop(label, axis = 1)
    Y_train = train[label]
    X_test = df[df[label].isna() == 1]
    X_test = X_test.drop(label, axis = 1)
    return X_train, Y_train, X_test

def remove_duplicates(df):
    df = df[df.duplicated() == 0]
    df.reset_index(drop = True, inplace = True)
    return df
        
def calc_iqr(df, col):
    Q3 = df[col].quantile(.75)
    Q1 = df[col].quantile(.25)
    IQR = Q3 - Q1
    threshold = 1.5 * IQR 
    lower_limit = Q1 - threshold 
    upper_limit = Q3 + threshold
    return lower_limit, upper_limit   
    
def dist_plot(df, cols):
    for col in cols:
        plt.figure() 
        print(col, " : ")
        sns.histplot(df[col], kde=True, bins=15)
        plt.title(f"Distribution of {col}")
        plt.show() 
        
def scat_plot(df, cols):
    for col in cols:
        plt.figure() 
        print(col, " : ")
        sns.scatterplot(y= df[col], x= df['Age'])
        plt.title(f"Distribution of {col}")
        plt.show()         
        
def box_plot(df, cols):
    for col in cols:
        plt.figure(figsize=(3,3))
        plt.ylabel(col)
        plt.boxplot(df[col])
        plt.show()
        
def transform_log(df, col):
    df[f'Log_transformed_{col}'] = np.log(df[col])
    return df

def reverse_one_hot(df, cat_cols):
    for col in cat_cols: 
        iKn = copy.deepcopy(df)
        dummyCols = [c for c in df.columns if col + "_" in c]
        print(dummyCols)
        df[col] = df[dummyCols].idxmax(axis = 1).str.replace(col + "_", "")
        df[col] = pd.to_numeric(df[col])
        df.drop(dummyCols, axis = 1, inplace = True)
      
def compute_partial_relation(df, label):
    cors = []
    for col in df.drop(label, axis = 1).columns:
        print(col + ":")
        print(pg.partial_corr(method="pearson", data = df, x = col, y = "Survived"))
    return cors
