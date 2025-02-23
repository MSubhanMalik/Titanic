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
X_train, Y_train, X_test = create_splits(df, "Age")
model = train_model(X_train, Y_train)
df.shape
df = impute_age(df, "Age", X_test, model)
df = remove_duplicates(df)

cols = ['SibSp', "Parch"]
box_plot(df, cols)
df['SibSp'].value_counts()
df['Parch'].value_counts()
dist_plot(df, cols)
upper_limit = np.percentile(df['SibSp'], 97.5)
upper_limit
valuesToBeCapped = np.where(df["SibSp"]>upper_limit)[0]
df.loc[valuesToBeCapped, "SibSp"] = 4
lower_limit, upper_limit = calc_iqr(df, "Parch")
lower_limit, upper_limit
upper_limit = np.percentile(df['Parch'], 98.5)
upper_limit
valuesToBeCapped = np.where(df["Parch"]>upper_limit)[0]
df.loc[valuesToBeCapped, "Parch"] = 4

df['Fare Per Person'] = df["Fare"]/(df["SibSp"]+df['Parch'] + 1)
df = transform_log(df, 'Fare Per Person')
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
print(data)

# Outliers Confirmed in Fare.
data = transform_log(data, 'Fare')
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
valuesToBeCapped
data.loc[valuesToBeCapped, "Age"] = cappedValue
data = data[data['Age']>=0]

# One Hot Encoding for Gender and Cities
new_data = copy.deepcopy(data)
new_data
new_data = pd.get_dummies(new_data, columns= ["Embarked", "Sex"])
new_data.shape
# 757 Rows

# Getting Data Ready for Isolation Forest
new_data = new_data.replace({True: 1, False: 0})
new_data['Pclass'] = new_data['Pclass'].replace({3: 1, 1:3})
cols = [col for col in new_data.columns if "Log" in col]
print(cols)
new_data.drop(cols, axis = 1, inplace = True)
new_data.shape
# 757 rows

# Multivariate Outliers 
anomaly_data = copy.deepcopy(new_data)
model = IsolationForest(n_estimators= 100, contamination=0.015, random_state=42)  # .804 .806
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

# Removing Anomaly Colums and adding based on knowledge
cols = ['Anomaly Score', "Anomaly"]
new_data.drop(cols, inplace = True, axis = 1)
# I am going to remove SibSp, and Fare per person as SibSp p value was against it and i think log will perform better as its results were more statistically significant
new_data['Total People'] = new_data["Parch"] + new_data["SibSp"] + 1
new_data = transform_log(new_data, "Fare Per Person")
new_data.drop(inplace=True, axis = 0, columns = ['Fare Per Person', 'SibSp'])
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

# Splitting the data
X_train = encoded_data.drop("Survived", axis = 1)
Y_train = encoded_data['Survived']
model = LGBMClassifier(metric = "rmse")
model.fit(X_train, Y_train)
y_train = model.predict(X_train)
print(np.sqrt(mse(Y_train, y_train)))

cv_scores = cross_val_score(model, X_train, Y_train, cv=5, scoring='accuracy')

print("Cross-Validation Accuracy:", np.mean(cv_scores))