

# Preprocessing the test data


# 1st Class: $30 – $512 (Expensive, luxurious cabins)
# 2nd Class: $13 – $73 (Mid-range, comfortable)
# 3rd Class: $7 – $40 (Budget, basic accommodations)

# Data is divided into mixed data types, therefore using MD or One SVM is not useful. We will use method like like quantile and isolation forests to detect outliers. We can use other algorithms and methods to fill in empty values. I have used LGBM Linear regression model as its results were multivariate and reasonable. Another thing is that I have applied Complete Case Analysis to remove features and rows with null values. In order to not effect the performance of Isolation forest, I will use One Hot Encoding. I will remove extreme outliers in uni variate

# After using box plot, I can clearly see, that most univariate outliers lie in Fare, Age, Parch and SibSp

# normal distribution is only in univariate Age column, only in that we can use z-score. Skewed is seen in Fare and others are categorical

# We will remove the extreme outliers and cap the normal outliers which we will believe to be removed by Isolation Forest

