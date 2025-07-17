import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

df=pd.read_csv("dataset/creditcard.csv")

df=df.drop(columns=['Time'])

# Separting features and target
X=df.drop(columns=['Class'])
y=df['Class']

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

# DBSCAN or Isolation Forest

dbscan=DBSCAN(eps=1.3, min_samples=5, n_jobs=-1)
# eps: Radius of neighbourhood
# min_samples: Minimum points in a cluster
# n_jobs is -1 which indicates "Noise" or "Outlier"
clusters=dbscan.fit_predict(X_scaled)

# Evaluate
print("Cluster labels and counts:", np.unique(clusters, return_counts=True))

# Use original y to evaluate
print(np.unique(clusters, return_counts=True))

# Map -1 to fraud, rest to legit
y_pred=np.where(clusters==-1,1,0)


print(confusion_matrix(y,y_pred))
print(classification_report(y,y_pred))

y_confidence = (clusters == -1).astype(float)

print("ROC-AUC:", roc_auc_score(y, y_confidence))
print("AUPRC:", average_precision_score(y, y_confidence))