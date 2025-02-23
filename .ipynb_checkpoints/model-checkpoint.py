import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor, LGBMClassifier
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif
import copy
import pingouin as pg
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_error as mse
from utility import *
from sklearn.model_selection import cross_val_score

# Information about the data
original = pd.read_csv("./titanic/train.csv")
test = pd.read_csv("./titanic/test.csv")
testLabel = pd.read_csv('./titanic/gender_submission.csv')
testDf = test.merge(testLabel, on="PassengerId")
df = copy.deepcopy(original)

df = process_data(df)
testDf = process_data(testDf)
X_train, Y_train, X_test = create_splits(df, "Age")
X_train_2, Y_train_2, X_test_2 = create_splits(testDf, "Age")
model = train_model(X_train, Y_train)
df.shape
testDf.shape
df = impute_age(df, "Age", X_test, model)
testDf = impute_age(testDf, "Age", X_test_2, model)
df = remove_duplicates(df)
# testDf = remove_duplicates(testDf)

cols = ['SibSp', "Parch"]
box_plot(df, cols)
df['SibSp'].value_counts()
df['Parch'].value_counts()
dist_plot(df, cols)
lower_limit, upper_limit = calc_iqr(df, "SibSp")
lower_limit, upper_limit
upper_limit = np.percentile(df['SibSp'], 97.5)
upper_limit
valuesToBeCapped = np.where(df["SibSp"]>upper_limit)[0]
valuesToBeCapped_test = np.where(testDf["SibSp"]>upper_limit)[0]
df.loc[valuesToBeCapped, "SibSp"] = 4
testDf.loc[valuesToBeCapped_test, "SibSp"] = 4
lower_limit, upper_limit = calc_iqr(df, "Parch")
lower_limit, upper_limit
upper_limit = np.percentile(df['Parch'], 98.5)
upper_limit
valuesToBeCapped = np.where(df["Parch"]>upper_limit)[0]
valuesToBeCapped_test = np.where(testDf["Parch"]>upper_limit)[0]
df.loc[valuesToBeCapped, "Parch"] = 4
testDf.loc[valuesToBeCapped_test, "Parch"] = 4
# df[df["Parch"]>upper_limit]

df['Fare Per Person'] = df["Fare"]/(df["SibSp"]+df['Parch'] + 1)
testDf['Fare Per Person'] = testDf["Fare"]/(testDf["SibSp"]+testDf['Parch'] + 1)
df = transform_log(df, 'Fare Per Person')
testDf = transform_log(testDf, 'Fare Per Person')
col = ['Log_transformed_Fare Per Person']
box_plot(df, col)
dist_plot(df, col)

# Outliers Confirmed in Fare per Person.
col = 'Log_transformed_Fare Per Person'
df.shape
lower_limit, upper_limit = calc_iqr(df, col)
data = df
lower_limit, upper_limit
data.shape
data = data[(data[col] < upper_limit)]
data.reset_index(drop = True, inplace = True)
testDf = testDf[(testDf[col] < upper_limit)]
testDf.reset_index(drop = True, inplace = True)
print(data)

# Outliers Confirmed in Fare.
data = transform_log(data, 'Fare')
testDf = transform_log(testDf, 'Fare')
col = 'Log_transformed_Fare'
data.shape
# lower_limit, upper_limit = calc_iqr(data, col)
# data.shape
# cappedValues = np.exp(data[col][(data[col] > upper_limit)])
# cappedValues
# valuesToBeCapped = np.where(data[col] > upper_limit)[0]
# np.exp(upper_limit)
# valuesToBeCapped
# data.loc[valuesToBeCapped, "Fare"] = np.exp(upper_limit)
data = data[data['Fare'] >= 4]
# testDf = testDf[testDf['Fare'] >= 4]
print(data[(data[col] > upper_limit)])

# Outliers Confirmed in Age.
col = ['Age']
box_plot(data, col)
data.shape
stanD = data['Age'].std()
meanD = data['Age'].mean()
cappedValue = meanD + 3*stanD
lower_limit, upper_limit = calc_iqr(data, "Age")
upper_limit
valuesToBeCapped = np.where(data[col] > 65)[0]
# valuesToBeCapped_test = np.where(testDf[col] > 65)[0]
valuesToBeCapped
data.loc[valuesToBeCapped, "Age"] = cappedValue
# testDf.loc[valuesToBeCapped_test, "Age"] = cappedValue

# One Hot Encoding for Gender and Cities
new_data = copy.deepcopy(data)
new_data
new_data = pd.get_dummies(new_data, columns= ["Embarked", "Sex"])
testDf = pd.get_dummies(testDf, columns= ["Embarked", "Sex"])
new_data.shape
# 757 Rows

# Getting Data Ready for Isolation Forest
new_data = new_data.replace({True: 1, False: 0})
testDf = testDf.replace({True: 1, False: 0})
new_data['Pclass'] = new_data['Pclass'].replace({3: 1, 1:3})
testDf['Pclass'] = testDf['Pclass'].replace({3: 1, 1:3})
cols = [col for col in new_data.columns if "Log" in col]
print(cols)
new_data.drop(cols, axis = 1, inplace = True)
testDf.drop(cols, axis = 1, inplace = True)
new_data.shape
# 757 rows

# Multivariate Outliers 
anomaly_data = copy.deepcopy(new_data)
model = IsolationForest(n_estimators= 100, contamination=0.03, random_state=42)
model.fit(anomaly_data)
new_data['Anomaly Score'] = model.decision_function(anomaly_data)
new_data["Anomaly"] = model.predict(anomaly_data)
# 757 rows

# Removing those outliers as they are multivariate and it will be difficult to adjust them again
new_data.shape
new_data = new_data[new_data["Anomaly"] != -1]
new_data.reset_index(drop = True, inplace= True)
new_data
cols = ["Sex", "Embarked"]
reverse_one_hot(new_data, cols)
reverse_one_hot(testDf, cols)

# Removing Anomaly Colums and adding based on knowledge
cols = ['Anomaly Score', "Anomaly"]
new_data.drop(cols, inplace = True, axis = 1)
# I am going to remove SibSp, and Fare per person as SibSp p value was against it and i think log will perform better as its results were more statistically significant
new_data['Total People'] = new_data["Parch"] + new_data["SibSp"] + 1
testDf['Total People'] = testDf["Parch"] + testDf["SibSp"] + 1
new_data = transform_log(new_data, "Fare Per Person")
testDf = transform_log(testDf, "Fare Per Person")
new_data.drop(inplace=True, axis = 0, columns = ['Fare Per Person', 'SibSp'])
testDf.drop(inplace=True, axis = 0, columns = ['Fare Per Person', 'SibSp'])
# Feature Selection 
# I am going to use Partial, Full Corelations and mutual information to determine which features I will use.
# Reversing one hot encoding to analyze the data
analyzed_data = copy.deepcopy(new_data)
pcCorr = compute_partial_relation(analyzed_data, "Survived")
print(analyzed_data.dtypes)
# Using MI - First Discretizing
# mi_data = analyzed_data
cols = ["Fare", "Log_transformed_Fare Per Person", "Age"]
discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
analyzed_data[cols] = discretizer.fit_transform(analyzed_data[cols])
mi_scores = mutual_info_classif(analyzed_data.drop("Survived", axis = 1), analyzed_data["Survived"], random_state=42)

mi_df = pd.DataFrame({'Feature': analyzed_data.drop("Survived", axis = 1).columns, 'MI Score': mi_scores})
print(mi_df)

corrFull = new_data.corr()
# pc_matrix = pcCorr.pivot(index="x", columns="y", values="r").fillna(1)
plt.figure(figsize=(8,6))
sns.heatmap(corrFull, annot=True, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()
# I have removed Fare Per Person and SibSp for better results and added Log Fare Per Person along with Total People that accounts for SibSp

# Performing the final touches 
# Encoding
cols = ["Sex", "Embarked"]
encoded_data = pd.get_dummies(data = new_data, columns = cols)
encoded_data_test = pd.get_dummies(data = testDf, columns = cols)

# Splitting the data
X_train = encoded_data.drop("Survived", axis = 1)
Y_train = encoded_data['Survived']
Y_test = encoded_data_test['Survived']
X_test = encoded_data_test.drop("Survived", axis = 1)
model = LGBMClassifier(metric = "rmse")
model.fit(X_train, Y_train)
y_train = model.predict(X_train)
y_test = model.predict(X_test)
print(np.sqrt(mse(Y_train, y_train)))

cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy:", np.mean(cv_scores))

# Preprocessing the test data


# 1st Class: $30 – $512 (Expensive, luxurious cabins)
# 2nd Class: $13 – $73 (Mid-range, comfortable)
# 3rd Class: $7 – $40 (Budget, basic accommodations)

# Data is divided into mixed data types, therefore using MD or One SVM is not useful. We will use method like like quantile and isolation forests to detect outliers. We can use other algorithms and methods to fill in empty values. I have used LGBM Linear regression model as its results were multivariate and reasonable. Another thing is that I have applied Complete Case Analysis to remove features and rows with null values. In order to not effect the performance of Isolation forest, I will use One Hot Encoding. I will remove extreme outliers in uni variate

# After using box plot, I can clearly see, that most univariate outliers lie in Fare, Age, Parch and SibSp

# normal distribution is only in univariate Age column, only in that we can use z-score. Skewed is seen in Fare and others are categorical

# We will remove the extreme outliers and cap the normal outliers which we will believe to be removed by Isolation Forest




















# Very Dangerous Method - It seems to 
# for col in cols:
#     stanD = 3*df[col].std()
#     mean = df[col].mean()
#     upperLimit = mean + stanD
#     df[col] = df[col].clip(upper = upperLimit)
        
# df
# box_plot(df, cols)
# Using Linear Regression to impute Values for age