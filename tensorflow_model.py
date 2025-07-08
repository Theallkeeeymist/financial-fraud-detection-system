import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import precision_recall_curve
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.version_utils import callbacks
from tensorflow.keras.callbacks import EarlyStopping

data=pd.read_csv("dataset/creditcard.csv")
data = data.drop(columns=['Time'])
print(data['Amount'].head())
print(data['Amount'].tail())

X=data.drop('Class', axis=1) #All the features that the model will use to make predictions
y=data['Class'] #The target column fraud or not
scaler=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_scaled=scaler.fit_transform(X_train)
x_test_scaled=scaler.transform(X_test)

#stratify produces similar ratio of data compared to original dataset i.e, if overall there is 0.2% of fraud then why will have similar percentage of fraud
smote=SMOTE(random_state=42)
X_train_sm,y_train_sm=smote.fit_resample(X_train_scaled,y_train)

model=Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1,activation='sigmoid') #Binary Classification
])

early_stop=EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

#
from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_sm),
    y=y_train_sm
)
class_weights_dict = dict(enumerate(class_weights))
#

model.fit(X_train_sm, y_train_sm, epochs=50, batch_size=32, validation_data=(X_test,y_test), callbacks=[early_stop],
          class_weight=class_weights_dict
         )

y_prob=model.predict(X_test).ravel()

precisions,recalls,thresholds=precision_recall_curve(y_test,y_prob)
# Find best threshold by maximizing F1 = 2 * (P*R)/(P+R)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]

print("Best threshold based on F1:", best_threshold)

# Now classify using this threshold
y_pred = (y_prob > best_threshold).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC: ", roc_auc_score(y_test, y_prob))
print("AUPRC: ", average_precision_score(y_test, y_prob))


import matplotlib.pyplot as plt

plt.plot(recalls, precisions)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.show()


# Scaling is done before SMOTE
# SMOTE neighbours ki range ke hisaab se interpolate karta hai synthetics data
# Aur jo value bada hoga woh dominate karega aur uske basis mein ban jayega data which is BAD data