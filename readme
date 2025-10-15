# Bank Subscription Prediction

##  my key decisions for this task
- i habitually prefer to work modularly so different parts of the pipeline can be individually debugged and run. 
- i have kept utils separate for use across projects. these in future would ideally have unit tests.
- usually i would split model inference into it's own script but for ease of submission have retained with training.
- i retained eda.ipynb to show some thinking into features (how to handle categoricals, assess missing values, feature engineering possibilties)
- i did not scale numerical variables as tree based methods handle these well generally, particularly as a first pass. this could be considered to reduce the problem space if there are a large number of outliers / spread in data.
- i one-hot encoded due to low cardinality of categorical variables, which means the feature space would not explode.
- i applied SMOTE to handle imbalance. undersampling can hide trends and oversampling can lead to biases, and balancing in model parameters alone did not produce an improved performance.
- tested 1 bagging and one boosting method. xgboost outperformed in first pass, so i ran randomisedCV to enhance eprformance. 
- i focussed on f1 to balance recall and precision and to set a cutoff for predictions - accuracy is not the best for imbalanced problems (represents great performance on the problems less interested class)

## Notes  
- Notebooks (`eda.ipynb`, `model_insights.ipynb`) are for exploration and insights only, not required for core model training.  

## improvements
- the model outputs dont all make sense (theres a heavy focus on contact month) - id want to understand this variable better (SMEs, check correlation with target, etc)
- more understanding of features generally
- feature importances is very slapdash - need to measure consistency of important features across models for stability
- feature impotance interpretation is not ideal - not easy to translate to insights. could put important features these into regression model to use coefficients as clear guidance on feature importance for example, or explore SHAP or other interpretations for clearer insights.



## Contents  
- **`utils.py`**  
  - key functions used across the scripts

- **`eda.ipynb`** (exploratory, dynamic, not essential to submission) 
  - retained to demonstrate some quick exploratory data analysis on data quality, class imbalance and features

- **`model.py`**  
  - data preprocessing (handling potential missing values, duplicates, encoding)  
  - feature engineering
  - SMOTE oversampling to balance classes  
  - model training using XGBoost with randomised hyperparameter search  
  - model evaluation and threshold optimisation  
  - model inference 
  - export prediction probabilities, prioritised prospect lists and feature importance

- **`model_insights.ipynb`** (not essential to submission)
  - quick analysis comparing current subscription rates with model-predicted prospects for insights for slides

## Usage  
- update file paths in `model.py` and `eda.ipynb` as needed.  
- run `model.py` to train the model, run inference, evaluate performance, and generate prospect lists.  

