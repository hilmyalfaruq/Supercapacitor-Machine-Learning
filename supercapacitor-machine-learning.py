import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_supercapacitors.csv')

data = data.drop(['no'], axis=1)
data.head()

import numpy as np
import seaborn as sns

sns.set(style="white")

columns = ['pw','ssa','pv','ps','idperig','npercent','opercent','capacitance']
corr = data[columns].corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.4, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.text(3.5, 3, "Cross-Correlation Between Features");
plt.text(3.8, 3.2, "Red = High, Blue = Low");

X1 = data[['pw','ssa','pv','ps','idperig','npercent','opercent']]
X2 = data[['ssa','pv','idperig','npercent','opercent']]
y = data[['capacitance']]

import numpy as np
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()
model1 = linear_regressor.fit(X1,y)

predictions1 = model1.predict(X1)

df11 = pd.DataFrame(predictions1) 

a = data[['capacitance']]
b1 = df11[[0]]

from scipy.stats import spearmanr

corr, _ = spearmanr(a, b1)
print('Spearmans correlation: %.3f' % corr)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(a, b1)
print('Mean Absolute Error: %.3f' % mae)

from sklearn.metrics import mean_squared_error
from math import sqrt

rmse = sqrt(mean_squared_error(a, b1))
print('Root Mean Square Error: %.3f' % rmse)

from scipy import stats
import seaborn as sns

sns.set_context('talk')
sns.set_style("darkgrid")

plt.figure(figsize=(8, 6))
plt.scatter(a, b1, color='red')
plt.plot( [0,500],[0,500] )
plt.xlabel("real capacitance (F/g)", labelpad=13)
plt.ylabel("predicted capacitance(F/g)", labelpad=13)
plt.title("Prediction using Linear Regression (LR)", y=1.015);
plt.text(400, 120, "LR", color='red')
plt.text(375, 80, "R: 0.577")
plt.text(350, 50, "MAE: 70.282")
plt.text(340, 10, "RMSE: 88.322")