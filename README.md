#storing data into a dataframe

import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris
from sklearn.inspection import permutation_importance
from sklearn.utils import Bunch
from sklearn.utils._bunch import Bunch

# Set the non-GUI backend for matplotlib
matplotlib.use('Agg')

# Define the missing function as a placeholder
def some_function_that_returns_bunch():
    # This is an example Bunch object.
    # Replace with actual data or logic as needed.
    return Bunch(importance_mean=[0.1, 0.2, 0.3])

#load weather data

weather_data = pd.read_csv("C:/Users/Aidan/Desktop/weather forecast/weather_forecast_data.csv")
print(weather_data)

weather_data.info()

#display summary statistics for numerical columns

print("\nSummary Statistics for Numerical Columns:")
print(weather_data.describe())


#checking for missing values
print("\nMissing Values in Each Column:")
print(weather_data.isnull().sum())

#Data Visualization and Analysis
fig, axes = plt.subplots(2, 3, figsize=(18,10))
fig.suptitle('Distribution of Numerical Features')

sns.histplot(weather_data['Temperature'], kde=True, ax=axes[0,0])
axes[0,0].set_title('Temperature')

sns.histplot(weather_data['Wind_Speed'], kde=True, ax=axes[0,1])
axes[0,1].set_title('Wind Speed')

sns.histplot(weather_data['Humidity'], kde=True, ax=axes[0,2])
axes[0,2].set_title('Humidity')

sns.histplot(weather_data['Pressure'], kde=True, ax=axes[1,0])
axes[1,0].set_title('Pressure')

sns.histplot(weather_data['Cloud_Cover'], kde=True, ax=axes[1,1])
axes[1,1].set_title('Cloud Cover')

# Plot Distribution of each numerical feature
weather_data.hist(figsize=(18,10))
plt.savefig('histogram_plot.png')
print("\nDistribution of Each Numerical Feature:")

#Distribution of Rain (target variable)

sns.countplot(x='Rain', data=weather_data, ax=axes[1,2])
axes[1,2].set_title('Rain Distribution')

plt.tight_layout()
plt.savefig('numerical_features_distribution.png')

#encode the target variable 'Rain'(convert  to binary: 'rain' as 1,  'no rain' as 0)
weather_data['Rain'] = weather_data['Rain'].map({'rain': 1, 'no rain': 0})

#separate features and target variable
X = weather_data.drop('Rain', axis=1)
y = weather_data['Rain']

#Get column names
columns = X.columns

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale the features to normalize teh data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test =  scaler.transform(X_test)

#initialize and train the Linear Regression

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

load_iris()

X,y = load_iris(return_X_y= True)

from sklearn.linear_model import LinearRegression
Model = LinearRegression()
Model.fit(X_train, y_train)

 #Making Predictions

predictions = Model.predict(X_test)
print(predictions)

matplotlib.use('Agg')

import matplotlib.pyplot as plt
predictions = Model.predict(X_test)
plt.scatter(predictions, y_test)
plt.savefig('predictions_vs_actual.png')

#feature importance with permutation importance

from sklearn.inspection import permutation_importance


#calculate permutation importance

perm_importance = permutation_importance(Model, X_test, y_test, n_repeats=10, random_state=42)

#calculate a DataFrame to visualize the results
feature_importance = pd.DataFrame(
    {'feature': columns, 'importance':
        perm_importance.importances_mean}
).sort_values(by='importance',
 ascending = False)

# Assuming perm_importance is your Bunch object
print(perm_importance.importances_mean)  # To see all available keys
print(dir(perm_importance))    # To see all available attributes  

#Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x= 'importance', y='feature', data=feature_importance)
plt.title('Permutation Feature Importance')
plt.xlabel('Mean Importance')
plt.ylabel('Feature')
plt.savefig('Agg.png')
