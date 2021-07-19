import numpy as np
import pandas as pd
import os
import sys
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# reading data
df = pd.read_csv('parkinsons.csv')

# print(df.shape)   (195, 24)

# extracting features(x) and labels(y)
features = df.copy().drop('status', axis=1).values[:, 1:]
labels = df['status'].values    # 147 True (1)   48 False (0)

# scale the features to between -1 and 1 to normalize them.
scaler = MinMaxScaler((-1, 1))

x = scaler.fit_transform(features)
y = labels

# split data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=7)

# label_encoder gives UserWarning cuz it gonna be removed next release and our data is already encoded
model = XGBClassifier(label_encoder=False)
model.fit(x_train, y_train)

predictions = model.predict(x_test)

score = accuracy_score(y_test, predictions)     # score around 94.87%

print(f'Accuracy: {round(score*100, 2)}')

# print(confusion_matrix(y_test, predictions))
#  [[ 6  1]
#  [ 1 31]]