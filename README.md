Team ID: TM-96   
Member 1: Attili Kireeti Aditya Kumar <br>
Member 2: Chua Zi Yang    
Member 2: Jeryl Neo Jia Le  

CTG Classification using Machine Learning:   
This project applies multiple machine learning algorithms to classify Cardiotocography (CTG) data into three fetal health states (Normal, Suspect, and Pathologic) based on physiological features collected from fetal monitoring.  

CTG.csv                     # Original raw dataset  
leaned_file.csv            # Cleaned and preprocessed dataset  
Final_DataCleaning.ipynb    # Data cleaning and preprocessing notebook  
ML.ipynb                    # Model training, evaluation, and comparison  
ctg_all_models_results.csv  # Results summary (auto-generated)  
ctg_all_models_predictions.csv # Test predictions for all models  
ctg_best_pipeline_<Model>.joblib # Saved best-performing model pipeline  

Objective:

To build a robust, explainable, and accurate classifier that automatically predicts the fetal state (NSP variable) using CTG readings. The model helps clinicians assess fetal well-being and detect abnormal patterns early, potentially reducing risks during pregnancy.

Methodology
1️. Data Preprocessing (Final_DataCleaning.ipynb)

Removed missing and duplicate values
Standardized feature scales
Handled outliers and irrelevant columns
Encoded categorical values (if any)

2️. Model Training (ML.ipynb)

We used the following models to train 
Random Forest
Logistic Regression
SVM (RBF Kernel)
XGBoost

Evaluation Metrics:

Cross-Validation Balanced Accuracy
Test Balanced Accuracy
F1-Macro Score
Accuracy
Each model’s confusion matrix and classification report are also generated.

3️. Model Selection

The model with the highest CV balanced accuracy (and strongest test F1) is automatically saved as: " ctg_best_pipeline_<ModelName>.joblib " 

You can use the following code to use our model to make new predictions: 
```
import pandas as pd
from joblib import load
def predict_nsp(df_raw: pd.DataFrame, model_path="ctg_best_pipeline_XGB.joblib"):
    model = load(model_path)
    return model.predict(df_raw)
```

Dependencies, download the following libraries to run our project using: 
```
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib
```









