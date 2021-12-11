#!/usr/bin/env python
# coding: utf-8

#Import libraries

import numpy as np 
import pandas as pd 

from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.experimental import enable_halving_search_cv
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, mutual_info_score
import xgboost as xgb
from sklearn.metrics import roc_curve,confusion_matrix, ConfusionMatrixDisplay,auc, RocCurveDisplay

#To suppress warnings, if any
import warnings
warnings.filterwarnings("ignore")

#Read data
data = 'heart_failure_clinical_records_dataset.csv'
df = pd.read_csv(data,delimiter=';')


#Feature Scaling
scaler = StandardScaler()

df_scaled = data.copy()
df_scaled.loc(axis=1)[numerical_cols.columns] = scaler.fit_transform(data.loc(axis=1)[numerical_cols.columns]) 



# Model comparison using Repeated Stratified K-Fold Cross Validation

RANDOM_STATE = 2 #Random state ensures that the splits we are tryin to generate are reproducible.

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=RANDOM_STATE)

feat_set = ['Leakage', 'Baseline']

# creating a list of models to evaluate.
models = [LogisticRegression(), 
          RandomForestClassifier(random_state=RANDOM_STATE), 
          XGBClassifier(verbosity=0, use_label_encoder = False, 
                        random_state=RANDOM_STATE, eval_metric='logloss'),
          GradientBoostingClassifier(random_state=RANDOM_STATE),
          SVC(kernel='sigmoid'),
          QuadraticDiscriminantAnalysis()]

model_names = [mod.__class__.__name__ for mod in models]

mod_cols = ['Name', 
            'Parameters',
            'Time']

df_mod = pd.DataFrame(columns=mod_cols)

for i in range(len(feat_set)):

    # Target variable feature considered.
    
    if (i==0):
        X = df_scaled.drop('DEATH_EVENT',axis=1)
    else:
        X = df_scaled.drop(['time','DEATH_EVENT'],axis=1)

     
    y = df_scaled['DEATH_EVENT']
    
    for j,model in enumerate(models):

        # Now, evaluating the models below.
        cv_results = cross_validate(model, X, y, cv=cv, scoring="f1", return_train_score = True)
        df_mod.loc[j + len(models)*i , 'Parameters'] = str(model.get_params())
        df_mod.loc[j + len(models)*i, 'Name'] = model.__class__.__name__
        df_mod.loc[j + len(models)*i, 'Time'] = cv_results['fit_time'].mean()
        df_mod.loc[j + len(models)*i, 'Train Accuracy'] = cv_results['train_score'].mean()
        df_mod.loc[j + len(models)*i, 'Test Score'] = cv_results['test_score'].mean()
        df_mod.loc[j + len(models)*i, 'feat_set'] = feat_set[i]






# To balance the dataset, we use Synthetic Minority Oversampling Technique.
counter = Counter(y) 

# transform the dataset
oversample = SMOTE()
X_sm, y_sm = oversample.fit_resample(X, y)
df_sm = X_sm.copy()
df_sm['DEATH_EVENT'] = y_sm
smote_counter = Counter(y_sm)

## Visualise the oversampling 
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax = [ax1, ax2]
for label, _ in counter.items():
    row_ix = np.where(y == label)[0]
    ax1.scatter(X.loc(axis=0)[row_ix]["platelets"],
                    [row_ix], 
                    X.loc(axis=0)[row_ix]["age"],
                    label=str(label),
                    alpha=0.75)
                    
for label_sm, _ in smote_counter.items():
    row_ix_sm = np.where(y_sm == label_sm)[0]
    ax2.scatter(X_sm.loc(axis=0)[row_ix_sm]["platelets"], 
                    [row_ix_sm],
                    X_sm.loc(axis=0)[row_ix_sm]["age"],
                    label=str(label_sm), alpha=0.75)

for axi in ax:
    axi.set_zlim(-2,2)
    axi.set_xlim(-2,2)
    axi.set_ylim(0,400)
    axi.set_xticks([-2,-1,0,1,2])
    axi.set_yticks([0,100,200,300,400])
    axi.set_zticks([-2,-1,0,1,2])
    
ax1.set_title("Original Data")
ax2.set_title("SMOTE Data")


# To evaluate the effect of Synthetic Minority Oversampling Technique on all our models
df_mod_sm = df_mod.copy()

for model in models:

    cv_results = cross_validate(model, X_sm, y_sm, cv=cv, scoring="f1", return_train_score = True)
    
    # Here,adding 1 to the max index instead of appending so I can pass everything as a dict()
    df_mod_sm.loc(axis=0)[df_mod_sm.index.values.max()+1] = {
            'Name':model.__class__.__name__,
            'Parameters':str(model.get_params()),
            'Time':cv_results['fit_time'].mean(),
            'Train Accuracy':cv_results['train_score'].mean(),
            'Test Score':cv_results['test_score'].mean(),
            'feat_set':'SMOTE'
             }

# Selecting appropriate features can result in improvements in model metrics
top_model_names = df_mod_sm.sort_values('feat_set', ascending=False).sort_values('Test Score', ascending=False)['Name'][:3].values
top_models = [m for m in models if m.__class__.__name__ in top_model_names]

# Initialize a DataFrame to contain the importances of each feature for each model
df_imp = pd.DataFrame(index=range(0,len(X_sm.columns)*len(top_models)), columns=['feature','model','importance'])

# len_feat will allow us to populate the features for each model
len_feat = int(len(X_sm.columns))

for i in range(len(top_models)):
    results = permutation_importance(top_models[i].fit(X_sm, y_sm), X_sm, y_sm, scoring="f1", 
                                n_repeats=10, n_jobs=None, 
                                random_state=RANDOM_STATE)   
    df_imp.loc[range(len_feat*i,len_feat*(i+1)),'importance'] = (results['importances_mean'])
    df_imp.loc[range(len_feat*i,len_feat*(i+1)),'model'] = top_models[i].__class__.__name__
    df_imp.loc[range(len_feat*i,len_feat*(i+1)),'feature'] = X_sm.columns



# Save model to disk
model_output_file = f'xgb_model.bin'

with open(model_output_file,'wb') as f_out:
    pickle.dump((poly,dv,model),f_out)