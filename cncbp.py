import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, 
auc, accuracy_score 
from xgboost import XGBClassifier 
# Load dataset (works in Colab) 
import kagglehub 
 
# Download latest version 
path = kagglehub.dataset_download("imakash3011/online-shoppers-purchasing
intention-dataset") 
 
print("Path to dataset files:", path) 
 
# Preprocess 
 
target = "Revenue" 
cat_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist() 
num_cols = df.select_dtypes(include=[np.number]).columns.tolist() 
num_cols.remove(target) 
 
preprocessor = ColumnTransformer([ 
    ('num', StandardScaler(), num_cols), 
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols) 
]) 
 
8 
 
X = df.drop(columns=[target]) 
y = df[target] 
# Train-test split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, 
random_state=42) 
 
# Train XGBoost model 
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
n_estimators=150) 
X_train_pre = preprocessor.fit_transform(X_train) 
X_test_pre = preprocessor.transform(X_test) 
 
model.fit(X_train_pre, y_train) 
y_pred = model.predict(X_test_pre) 
y_prob = model.predict_proba(X_test_pre)[:, 1] 
# Print Accuracy 
acc = accuracy_score(y_test, y_pred) 
print(f"🎯 Model Accuracy: {acc:.2f}") 
# 1️⃣ Feature Importance Plot 
importances = model.feature_importances_ 
features = preprocessor.get_feature_names_out() 
imp_df = pd.DataFrame({'Feature': features, 'Importance': 
importances}).sort_values('Importance', ascending=False).head(15) 
 
plt.figure(figsize=(10,6)) 
sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis') 
plt.title("Top 15 Important Features (XGBoost)") 
plt.show() 
 
# 2️⃣ Confusion Matrix 
 
cm = confusion_matrix(y_test, y_pred) 
9 
 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Purchase', 
'Purchase']) 
disp.plot(cmap='Blues') 
plt.title("Confusion Matrix - XGBoost") 
plt.show() 
 
# 3️⃣ ROC Curve 
fpr, tpr, _ = roc_curve(y_test, y_prob) 
roc_auc = auc(fpr, tpr) 
 
plt.figure(figsize=(6,5)) 
plt.plot(fpr, tpr, color='orange', lw=2, label=f"AUC = {roc_auc:.2f}") 
plt.plot([0,1], [0,1], color='navy', linestyle='--') 
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.title("ROC Curve - XGBoost Model") 
plt.legend(loc="lower right") 
plt.show() 