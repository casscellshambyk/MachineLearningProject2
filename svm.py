import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from time import time  # for timing classifications


dataset = pd.read_csv("datasets_180_408_data.csv")
dataset.columns
dataset.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

X = dataset.iloc[:,1:32].values
y = dataset.iloc[:,0].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

sns.pairplot(dataset,vars=['radius_mean','texture_mean','area_mean','smoothness_mean'],hue='diagnosis')
plt.ioff()

model = SVC()

polymodel = SVC(kernel='poly', degree=8)

start_time = time()
model.fit(X_train, y_train)
end_time = time()
time_elapsed = end_time - start_time
print(pd.Timedelta(time_elapsed, 'S'))
start_time_poly = time()
polymodel.fit(X_train, y_train)
end_time_poly = time()
time_elapsed_poly = end_time_poly - start_time_poly
print(pd.Timedelta(time_elapsed_poly, 'S'))
test_predictions = model.predict(X_test)
poly_predictions = polymodel.predict(X_test)

c_matrix = confusion_matrix(y_test, test_predictions)
c_matrix_poly = confusion_matrix(y_test, poly_predictions)

# sns.heatmap(c_matrix,annot=True)
# plt.ioff()
print("Linear Model")
print(classification_report(y_test, test_predictions))
print("Polynomial Kernal Model")
print(classification_report(y_test, poly_predictions))
