
# coding: utf-8

# In[3]:


import numpy as np 
import pandas as pd 

train_data = pd.read_csv("./SAN FRANCISCO/train.csv", index_col=False)  
pd.set_option('display.max_columns', None)
target = train_data["Category"].unique()

test_data = pd.read_csv("./SAN FRANCISCO/test.csv")

data_dict = {}
count = 1
for data in target:
    data_dict[data] = count
    count = count + 1
train_data["Category"] = train_data["Category"].replace(data_dict)


train_data['Dates'] = pd.to_datetime(train_data.Dates)

train_data['HORA'] = train_data['Dates'].dt.strftime('%H').astype(int)
train_data['PERIODO'] = np.around(train_data['HORA']/6).astype(int)
train_data['DAYOFYEAR'] = train_data['Dates'].apply(lambda x: x.dayofyear)

data_week_dict= {
    "Monday": 1,
    "Tuesday":2,
    "Wednesday":3,
    "Thursday":4,
    "Friday":5,
    "Saturday":6,
    "Sunday":7
}
train_data["DayOfWeek"] = train_data["DayOfWeek"].replace(data_week_dict)
test_data["DayOfWeek"] = test_data["DayOfWeek"].replace(data_week_dict)


train_data["X"] = (train_data["Y"]/180)
test_data["X"] = (test_data["Y"]/180)
train_data["Y"] =(train_data["Y"]/180)
test_data["Y"] = (test_data["Y"]/180)

district = train_data["PdDistrict"].unique()
data_dict_district = {}
count = 1
for data in district:
    data_dict_district[data] = count
    count+=1
train_data["PdDistrict"] = train_data["PdDistrict"].replace(data_dict_district)
test_data["PdDistrict"] = test_data["PdDistrict"].replace(data_dict_district)

columns_train = train_data.columns 

columns_test = test_data.columns

cols = columns_train.drop("Resolution")

train_data_new = train_data[cols]

corr = train_data_new.corr()

skew = train_data_new.skew()

features = ["DAYOFYEAR","PERIODO","DayOfWeek", "PdDistrict", "X", "Y"]
X = train_data[features]
y = train_data["Category"]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state = 0)


# # MLPClassifier

# In[4]:


from sklearn.neural_network import MLPClassifier
modelmlp = MLPClassifier()
modelmlp.fit(X_train,y_train)


# In[5]:


y_predictMLPC = modelmlp.predict(X_test)


# In[6]:


from sklearn.metrics import accuracy_score, r2_score
print r2_score(y_test,y_predictMLPC)
print accuracy_score(y_test,y_predictMLPC)


# # GradientBoostingClassifier

# In[ ]:


from sklearn import ensemble
clf = ensemble.GradientBoostingClassifier()
clf.fit(X_train, y_train)


# In[ ]:


y_prediGBC = clf.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, r2_score
print r2_score(y_test,y_prediGBC)
print accuracy_score(y_test,y_prediGBC)


# # LSTM

# In[ ]:


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# In[ ]:


model = Sequential()
model.add(LSTM(3, input_shape=(1, 4)))
model.add(Dense(1))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


X_train = np.array(X_train)
y_train = np.array(y_train)


# In[ ]:


X_train_lstm = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))


# In[ ]:


model.fit(X_train_lstm, y_train, epochs=100, batch_size=1, verbose=2)


# In[ ]:


y_preditoLSTM = model.predict(X_test_lstm)


# In[ ]:


from sklearn.metrics import accuracy_score, r2_score
print r2_score(y_test,y_preditoLSTM)
print accuracy_score(y_test,y_preditoLSTM)


# # RandomForest

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators = 250, criterion = 'gini',max_depth = 10)
RF.fit(X_train, y_train)
y_predict_rf = RF.predict(X_test)


# In[ ]:


from sklearn.metrics import accuracy_score, r2_score
print r2_score(y_test,y_predict_rf)
print accuracy_score(y_test,y_predict_rf)


# # KNeighborsClassifier

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_treino, y_treino)
y_predict_Rnn = knn.predict(X_teste)


# In[ ]:


from sklearn.metrics import accuracy_score, r2_score
print r2_score(y_test,y_predict_Rnn)
print accuracy_score(y_test,y_predict_Rnn)

