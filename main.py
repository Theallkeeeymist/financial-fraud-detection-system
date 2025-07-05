import pandas as pd
from sklearn.preprocessing import StandardScaler #Using this for feature scaling'
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, average_precision_score, \
    classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

data=pd.read_csv("dataset/creditcard.csv")

# print(data.head)
# print('\n')
# print(data.shape)
# print('\n')
# print(data.columns)
# print('\n')
# print(data['Class'].value_counts())
# Drop 'Time' column if not already
data = data.drop(columns=['Time'])

# print(data['Amount'].head()) #RAW DATA IS SUITABLE FOR TREE BASED MODEL SINCE THE DO NOT REQUIRE SCALING AT ALL.

scaler=StandardScaler()  #TO BE USED WHEN REGRESSION OR SVM OR KNN TYPE MODELS ARE USED.
data['Amount']=scaler.fit_transform(data[['Amount']])
print(data['Amount'].head())

# scaler=MinMaxScaler()   TO BE USED WHEN FEATURES ARE DIFFERENTLY SCALED AND YOU WANT UNIFORMITY.
# data['Amount']=scaler.fit_transform(data[['Amount']])
# print(data['Amount'].head())
X=data.drop('Class', axis=1) #All the features that the model will use to make predictions
y=data['Class'] #The target column fraud or not

print(X.head())
print('\n')
print(y.tail())

X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#stratify produces similar ratio of data compared to original dataset i.e if overall there is 0.2% of fraud then why will haeve similar percentage of fraud
smote=SMOTE(random_state=42)
X_train_sm,y_train_sm=smote.fit_resample(X_train,y_train)
# model_logistic=LogisticRegression(class_weight='balanced')
# model_logistic.fit(X_train, y_train)

# model_logistic=LogisticRegression()
# model_logistic.fit(X_train_sm, y_train_sm)
# y_pred=model_logistic.predict(X_test)
# y_prob=model_logistic.predict_proba(X_test)[:, 1]

model_tree=RandomForestClassifier(
    n_estimators=200, #Number of tree
    # class_weight='balanced', #Handle fraud vs legit imbalance
    random_state=42,
    n_jobs=-1 #Use all CPU Cores
)
model_tree.fit(X_train_sm, y_train_sm)

y_pred=model_tree.predict(X_test)
y_prob=model_tree.predict_proba(X_test)[:,1]

print("🔹 Logistic Regression (class_weight='balanced')")
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("AUPRC:", average_precision_score(y_test, y_prob))


# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm=confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest - Confusion Matrix')
plt.show()