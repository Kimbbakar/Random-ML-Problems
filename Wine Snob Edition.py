import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score	
from sklearn.externals import joblib
 
data = pd.read_csv('winequality.csv',sep=';')

Y = data.quality
X = data.drop('quality',axis = 1)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state=123,stratify = Y )