import sys
import datetime
import pandas as pd
import numpy as np
import math
import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score





def load_data(data_filepath):
    
    """
    Load data from flat file 
    
    Args:
        data_filepath: path where flat csv file is located
    
    Returns:
        df: dataframe of flat file
    """
    df = pd.read_csv(data_filepath, parse_dates = ['became_member_on'])
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    
    return df

def feature_engineering(df):
    
    
    """
    Performs remaining data preprocessing and feature engineering to transform data to format ready for model building
    
    Args:
        df: loaded dataframe
    
    Returns:
        df: clean dataframe ready for model building (or intermediary upsampling step)
    
    """
    
    df = df.loc[df.event == 'offer received']
      
    df = df.dropna(subset=['income', 'gender'])
    
    df['target_class'] = 0 #CN
    df.loc[(df.treatment == 0) & (df.outcome ==1),'target_class'] = 1 # Control Responders
    df.loc[(df.treatment == 1) & (df.outcome  ==0),'target_class'] = 2 # treatment Non-Responders
    df.loc[(df.treatment == 1) & (df.outcome  ==1),'target_class'] = 3 # treatment Responders
    
    df.to_csv('data/clean_df.csv', index=False)
    
    df = df.drop(columns=['id','event', 'time', 'became_member_on', 'amount', 'offer_id', 'reward', 'max_time' ], axis=1)
    
    cat_vars = df.select_dtypes(include=['object']).copy().columns
  
   
    for var in cat_vars:
        # for each cat add dummy var, drop original column
        df = pd.concat([df.drop(var, axis=1), pd.get_dummies(df[var], prefix=var, prefix_sep='_', drop_first=True)], axis=1)

    return df

def upsample(df, column, majority_value, minority_value):
    
    """
    
    Upsamples dataframe to address class imbalance
    
    Args:
        df: dataframe 
        column: string feature that contains class imbalance
        majority_value: value of feature that is majority
        minority_value: value of feature that is minority
    
    Returns:
        df_upsampled: Dataframe with upsampling. Note that this should only be for training data.
    """
    
    # Up-sample Minority Class approach from Elite Data Science 
    # https://elitedatascience.com/imbalanced-classes


    # Seperate majority and minority classes
    df_majority = df[df[column] == majority_value]
    df_minority = df[df[column] == minority_value]
    
    majority_n_samples = df[column].value_counts()[majority_value]
    
    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                replace=True,
                                n_samples=majority_n_samples,
                                random_state=42)

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    return df_upsampled

def upsample_train(train_df):
    
    """
    Upsamples training data and returns X and y dataframes for build_model arguements
    
    Args:
        train_df: dataframe containing training data.
    
    Returns:
        X_train: dataframe containing features to be inputted for classifier model
        y_train: dataframe containing 'target_class'
    
    """
    train_df_upsample = upsample(train_df, 'treatment', 0, 1)
    X_train = train_df_upsample.drop(columns=['treatment', 'outcome','target_class'], axis=1)
    y_train = train_df_upsample.target_class
    
    
    
    return X_train, y_train


def build_model():
    
    """
    Builds pipeline and use grid search to perform classification
    
    Returns:
        cv: GridSearchCV pipeline with best parameters for the model
    """
    pipeline = Pipeline([
        ('clf', RandomForestClassifier())
        ])

    parameters = {
            'clf__n_estimators': [50, 100, 200], # number of trees in forest
            'clf__min_samples_split': [2, 3, 4], # minimum number of samples required for split to occur
            'clf__max_depth' : [4,5,6], # maximum depth of the tree; min samples_split supercedes in cases where max_depth isn't sufficient
            'clf__bootstrap': [True, False] # determine whether whole dataset should be used to build each tree

    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv
    

def evaluate_model(model, X_test, Y_test):
    """
    Evaluates model by providing individual category and summary metrics of model performance
    Args:
        model: RandomForestClassifier model
        X_test: subset of X values withheld from the model building process
        Y_test: subset of Y values witheld from the model building process and used to evaluate model predictions
        
    Returns:
        report: classification report with evaluation metrics (f1, precision, recall, support)
    """
     
    y_pred = model.predict(X_test)

    report = classification_report(y_pred, Y_test, labels = [0,1,2,3], target_names= ['CN', 'CR', 'TN', 'TR'], output_dict=True)

    print(report)
    
    return report


def save_report(report):

    """
    Loads classification report to csv file
    Args:
        report: classification report returned from evaluate_model function
        
    Returns:
        report_df: save dataframe as a csv at specified file path
    """

    report_df = pd.DataFrame(report).transpose()

    report_df.columns = ['f1', 'precision', 'recall', 'support']

    report_df['labels'] = report_df.index

    report_df = report_df[['labels','f1', 'precision', 'recall', 'support']]
    
    report_df.to_csv('data/report.csv', index=False)


    return report_df



def calc_uplift(model, df):
    
    """
    Calculates uplift scores for entire dataframe
    
    Args:
        model: RandomForestClassifier model
        df: clean dataframe that was returned form feature_engineering() function
    
    Returns:
        df: dataframe with uplifts scores for each event appended as series, titled 'uplift_score', as last column of dataframe
        
    """
    df_model = df.drop(['outcome', 'treatment','target_class'],axis=1)
    overall_proba = model.predict_proba(df_model)

    df['proba_CN'] = overall_proba[:,0] 
    df['proba_CR'] = overall_proba[:,1] 
    df['proba_TN'] = overall_proba[:,2] 
    df['proba_TR'] = overall_proba[:,3]

    df['uplift_score'] = df.eval('proba_CN + proba_TR - proba_TN - proba_CR')
    
    df.to_csv('data/uplift_df.csv', index=False)
    
    return df

    
def calc_cumulative_gains(df):
    
    """
    Creates dataframe for cumulative gain chart
    
    Args:
        df: dataframe containing uplift scores
    
    Returns:
        df: dataframe containing cumulative gain % values for uplift model corresponding with % of population
    """
    df['TR_Value'] = 3
    
    rows = []
    for group in np.array_split(df.sort_values(by='uplift_score', ascending=True), 100):
        score = accuracy_score(group['target_class'].tolist(),
                                                   group['TR_Value'].tolist(),
                                                   normalize=False)

        rows.append({'NumCases': len(group), 'NumCorrectPredictions': score})

    lift = pd.DataFrame(rows)
    
    
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
    

    lift.to_csv('data/cum_gains_df.csv', index=False)

    return lift


def main():
    if len(sys.argv) == 2:
        data_filepath = sys.argv[1]
        
        print('Loading data...')
        df = load_data(data_filepath)
        
        print('Feature engineering...')
        clean_df = feature_engineering(df)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        train_df, test_df = train_test_split(clean_df, test_size=0.3, random_state=42)
        
        print('Upsampling...')
        X_train, y_train = upsample_train(train_df)
        
        X_test = test_df.drop(columns=['treatment', 'outcome','target_class'], axis=1)
        y_test = test_df.target_class
        
        print('Building model...')
        model = build_model()
        
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        
        print('Evaluating model...')
        report = evaluate_model(model, X_test, y_test)
        
        print('Saving report...')
        save_report(report)
        
        print('Calculating uplift on test data...')
        uplift_df = calc_uplift(model, clean_df)
        
        print('Calculating cumulative gains...')
        cumalative_gains_df = calc_cumulative_gains(uplift_df)
        
    else:
        print('One arguement, data filepath, required')
        
if __name__ == '__main__':
    main()
