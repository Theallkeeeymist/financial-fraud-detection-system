import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam

data=pd.read_csv("dataset/creditcard.csv")
data = data.drop(columns=['Time'])
print(data['Amount'].head())
print(data['Amount'].tail())

X=data.drop('Class', axis=1) #All the features that the model will use to make predictions
y=data['Class'] #The target column fraud or not

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#stratify produces similar ratio of data compared to original dataset i.e if overall there is 0.2% of fraud then why will have similar percentage of fraud
smote=SMOTE(random_state=42)
X_train_sm,y_train_sm=smote.fit_resample(X_train,y_train)

model=Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1,activation='sigmoid') #Binary Classification
])

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

model.fit(X_train_sm, y_train_sm, epochs=50, batch_size=32,validation_data=(X_test,y_test))

y_prob=model.predict(X_test).ravel()
y_pred=(y_prob>0.5).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_prob))
print("AUPRC: ", average_precision_score(y_test, y_prob))