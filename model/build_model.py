import sys
import pandas as pd
import numpy as np
import math
import json


import xgboost as xgb
from xgboost import XGBClassifier

#from pylift import TransformedOutcome

from itertools import combinations
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


#import scikitplot as skplt
#import causallift
#from causallift import CausalLift



def load_data(data_filepath):
    df = pd.read_csv(data_filepath, parse_dates = ['became_member_on'])
    
    return df

def feature_engineering(df):
      
    df = df.dropna(subset=['income', 'gender'])
    
    df = df.drop(columns=['id','event', 'time', 'became_member_on', 'amount', 'offer_id', 'reward', 'max_time' ], axis=1)
    
    cat_vars = df.select_dtypes(include=['object']).copy().columns
  
   
    for var in cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)
    
    df['target_class'] = 0 #CN
    df.loc[(df.treatment == 0) & (df.outcome ==1),'target_class'] = 1 # Control Responders
    df.loc[(df.treatment == 1) & (df.outcome  ==0),'target_class'] = 2 # treatment Non-Responders
    df.loc[(df.treatment == 1) & (df.outcome  ==1),'target_class'] = 3 # treatment Responders
    
    X = df.drop(columns=['treatment', 'outcome','target_class'], axis=1)
    y = df.target_class
    
    return df, X, y


def build_model():
    model = xgb.XGBClassifier()
    
    return model
    

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates model by providing individual category and summary metrics of model performance
    Args:
        model: MultiOutputClassifier model
        X_test: subset of X values withheld from the model building process
        Y_test: subset of Y values witheld from the model building process and used to evaluate model predictions
        category_names: labels for model
    Returns:
        report: classification report with evaluation metrics (f1, precision, recall, support)
    """
     
    y_pred = model.predict(X_test)

    report = classification_report(y_pred, Y_test, labels = [0,1,2,3], target_names= ['CN', 'CR', 'TN', 'TR'], output_dict=True)

    print(report)
    
    return report


def save_report(report, report_filepath):

    """
    Loads classification report to csv file
    Args:
        report: classification report returned from evaluate_model function
        report_filepath: path for where to save report
    Returns:
        report_df: save dataframe as a csv at specified file path
    """

    report_df = pd.DataFrame(report).transpose()

    report_df.columns = ['f1', 'precision', 'recall', 'support']

    report_df['labels'] = report_df.index

    report_df = report_df[['labels','f1', 'precision', 'recall', 'support']]

    report_df.to_csv(report_filepath, index=False)


    return report_df



def calc_uplift(model, df, filepath):
    df_model = df.drop(['outcome', 'treatment','target_class'],axis=1)
    overall_proba = model.predict_proba(df_model)

    df['proba_CN'] = overall_proba[:,0] 
    df['proba_CR'] = overall_proba[:,1] 
    df['proba_TN'] = overall_proba[:,2] 
    df['proba_TR'] = overall_proba[:,3]

    df['uplift_score'] = df.eval('proba_CN + proba_TR - proba_TN - proba_CR')
    
    df.to_csv(filepath, index=False)
    
    return df

    
def calc_cumulative_gains(df, filepath):
    df['is_target_responder'] = 3
    
    rows = []
    for group in np.array_split(df.sort_values(by='uplift_score', ascending=False), 100):
        score = accuracy_score(group['target_class'].tolist(),
                                                   group['is_target_responder'].tolist(),
                                                   normalize=False)

        rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})

    lift = pd.DataFrame(rows)
    
    #lift['%responders'] = lift['NumCorrectPredictions']/ np.sum(lift.NumCorrectPredictions)
    
    correct_total = []
    total = 0

    for i in range(100):
        increment = lift.NumCorrectPredictions[i] * 1
        total += increment
        correct_total.append(total)

    
    lift['cum_correctpredictions'] = correct_total
    
    population_total = []
    pop_num = 0

    for i in range(100):
        pop_num += lift.NumCases[i] *1
        population_total.append(pop_num)

    lift['cum_pop'] = population_total
    
    lift['%predictors'] = lift.cum_correctpredictions/np.sum(lift.NumCorrectPredictions)

    lift['%population'] = lift.cum_pop/np.sum(lift.NumCases)
    
    #plt.plot (test_x, test_y)
    
    lift.to_csv(filepath, index=False)

    return lift


def main():
    if len(sys.argv) == 5:
        data_filepath, report_filepath, uplift_model_filepath, cum_gains_filepath = sys.argv[1:]
        
        print('Loading data...')
        df = load_data(data_filepath)
        
        print('Feature engineering...')
        clean_df,X,y = feature_engineering(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        report = evaluate_model(model, X_test, y_test)
        
        print('Saving report...')
        save_report(report, report_filepath)
        
        print('Calculating uplift on test data...')
        uplift_df = calc_uplift(model, clean_df, uplift_model_filepath)
        
        print('Calculating cumulative gains...')
        cumalative_gains_df = calc_cumulative_gains(uplift_df, cum_gains_filepath)
        
    else:
        print('Provide the first arguement as the data_filepath '              'and second arguement as the report_filepath. The '              'third arguement is the uplift_model filepath and '              'fourth arguement is the cum_gains_filepath.')
        
if __name__ == '__main__':
    main()

