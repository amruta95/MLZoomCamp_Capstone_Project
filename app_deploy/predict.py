#Import libraries
import pandas as pd
import xgboost as xgb
import pickle
from flask import Flask, request, jsonify


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_val_score, cross_validate
from sklearn.feature_extraction import DictVectorizer



#Parameters
model_file = 'xgb_model.bin'
threshold = 0.32

print("Loading model from file on disk")
with open(model_file,'rb') as f_in:
    poly, dv,model = pickle.load(f_in)



def predict_death_event(patient):
    df_patient = pd.DataFrame() #Create dataframe to hold the patient info.
    df_patient = df_patient.append(patient,ignore_index=True)

    
    cols = list(df_patient.columns.values)

     #From the input data, drop 'time' as this is not a useful feature.

    if 'time' in cols:
        del df_patient['time']


# Synthetic Minority Oversampling Technique

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

#Selecting appropriate features can result in improvements in model metrics
top_model_names = df_mod_sm.sort_values('feat_set', ascending=False).sort_values('Test Score', ascending=False)['Name'][:3].values
top_models = [m for m in models if m.__class__.__name__ in top_model_names]

# Initialize a DataFrame to contain the importances of each feature for each model
df_imp = pd.DataFrame(index=range(0,len(X_sm.columns)*len(top_models)), columns=['feature','model','importance'])

# len_feat will allow us to populate the features for each model
len_feat = int(len(X_sm.columns))

RANDOM_STATE = 2
for i in range(len(top_models)):
    results = permutation_importance(top_models[i].fit(X_sm, y_sm), X_sm, y_sm, scoring="f1", 
                                n_repeats=10, n_jobs=None, 
                                random_state=RANDOM_STATE)   
    df_imp.loc[range(len_feat*i,len_feat*(i+1)),'importance'] = (results['importances_mean'])
    df_imp.loc[range(len_feat*i,len_feat*(i+1)),'model'] = top_models[i].__class__.__name__
    df_imp.loc[range(len_feat*i,len_feat*(i+1)),'feature'] = X_sm.columns


#feature importance using SHAP 
shap.initjs()
explainer = shap.TreeExplainer(top_models[2].fit(X_sm, y_sm))
shap_values = explainer.shap_values(X_sm)
shap.summary_plot(shap_values, features=X_sm, feature_names=X_sm.columns)


feats = []

df_mod_sm_f = df_mod_sm[df_mod_sm['Name'].isin(top_model_names)].copy()

for j in [1,6]:
    feats.append(X_sm.columns[np.argsort(np.abs(shap_values).mean(0))][::-1][:j+1].values)

for i,feat in enumerate(feats):    
    for model in top_models:
        cv_results = cross_validate(model, X_sm.loc(axis=1)[feat], y_sm, cv=cv, 
                                    scoring="f1", return_train_score = True)

        # Adding 1 to the max index instead of appending so I can pass everything as a dict()
        df_mod_sm_f.loc(axis=0)[df_mod_sm_f.index.values.max()+1] = {
                'Name':model.__class__.__name__,
                'Parameters':str(model.get_params()),
                'Time':cv_results['fit_time'].mean(),
                'Train Accuracy':cv_results['train_score'].mean(),
                'Test Score':cv_results['test_score'].mean(),
                'feat_set': f"SMOTE {len(feat)}-Feature"
                 }
fig = px.bar(data_frame = df_mod_sm_f.sort_values('Test Score', ascending=True),
             x="Name", y="Test Score", color="feat_set", barmode="group",
             color_discrete_sequence=px.colors.qualitative.D3,
             template = "plotly_white")
fig.show()


y_patient_pred = model.predict(dpaient)
subscription = (y_patient_pred > threshold)

return y_patient_pred, subscription

app = Flask('subscription')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    y_paient_pred, subscription = predict_term_subscription(patient)

    if subscription:
        description = "Prediction: Patient will die due to Heart Failure"
    else:
        description = "Prediction: Patient will no die due to Heart Failure"


 
    result = {
            'subscription_probability': float(y_customer_pred),
            'subscription': bool(subscription),
            'description': description
            }

    return jsonify(result)

    if __name__ == "__main__":
   app.run(debug=True, host='0.0.0.0', port=9292)