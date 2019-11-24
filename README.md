# starbucks-uplift
Build an uplift model with machine learning pipeline to prioritize promotional outreach

### Summary
This project contains simulated Starbucks data of promotional offers and purchasing behavior. 

The deliverable is a dashboard that contains visualizations, most notably the cumulative gains chart, which aid busines users in their assessment of the uplift model performance and related target segment characteristics. More details regarding the deliverable are below in the Results section.

The repository contains working code for running an ETL pipeline, ML pipeline, and Flask app locally. Instructions are below.

### Instructions:
1. Run the following commands in the project's root directory to set up your data and model.

    - To run ETL pipeline that cleans data and stores processed data in a csv 
        `python data/process_data.py data/processed_data.py`
    - To run ML pipeline that trains classifier and outputs various csv files in data folder 
        `python model/build_model.py data/processed_data_11202019.csv.csv`

Please note that the process_data.py file currently takes an (excessively) long time to run ~5 hours due to the set_treatment_outcome function's iterative approach. I've included a sample output entitled 'processed_data_11202019.csv.csv'

2. Run the following command to run your web app.
    `python myapp.py`

3. Go to http://0.0.0.0:3001/


### Problem Statement

An uplift model seeks to predict the incremental value from sending a promotion.

The objective is to create an uplift model that scores and prioritizes events where offer is both viewed and a transaction is completed within the offer duration.

A classification model will be developed to predict the results of an offer sent, which is termed as an "event" throughout this documentation.

The classifications will be four target segments, which are the following:

1) Control Non-Responders (Treatment: 0, Outcome: 0)

2) Control Responders (Treatment: 0, Outcome: 1)

3) Treatment Non-Responders (Treatment: 1, Outcome: 0)

4) Treatment Responders (Treatment: 1, Outcome: 1)

The goal is to prioritize Treatment Responders in the outreach. This will be evaluated with the use of a cumulative gains chart, which will be discussed in more detail below. 

Why would we not be prioritizing Control Responders as well? Well, those are events that would be termed "sure things" in the marketing world. We're focused on the events where there is incremental value to be gained.

### Results

![](https://github.com/taylorplumer/starbucks-uplift/blob/master/img/cumulative_gains_chart.png)

The cumulative gains chart above shows performance of prioritizing the Treatment Responders with the uplift model compared to a random choice. 

Performance could definitely be improved as measured by the area under the curve. A more predictive model would be pushed out more to the upper left. 

The 'days_as_member' feature is indicative of Treatment Responder with the median value being higher in the Treatment Class than the other classes.

![](https://github.com/taylorplumer/starbucks-uplift/blob/master/img/Average_Conversion_Rate_by_OfferType_and_Channel.png)

On average, the discount was the most successful offer type in terms of conversion. The channels all performed relatively similar in terms of conversion with the exception of the web, which underperformed relative to the rest for all offer types.

__Next Steps__

1.  Other classification methods that take advantage of boosting, such as AdaBoost or XGBoos, could be employed to improve the performance of the uplift model. Currently the machine learning pipeline is limited to grid search on parameters for only a Random Forest classifier.
2.  An alternative approach to feature engineering the 'treatment' and 'outcome' featueres. Currently, the set_outcome_treatment() function is used in the process_data.py file but is excessively slow. It takes approximately ~5 hours to run. Identifying functions native to Pandas and Numpy that more effectively utilize vector operations could be adopted to improve performance. 


### Important Files:

1.  data/process_data.py: ETL script to clean and saves data to csv file
2.  models/build_model.py: builds, trains, evaluates, and  outputs results of machine learning classifier prediction
3.  data/visualizations.py: returns plotly visualizations for Flask application
4.  myapp.py: this file is used to run the Flask application

### Additional Files of note:

__Analysis notebooks__

1.  data/Exploratory_Data_Analysis.ipynb: jupyter notebook that explores initial data files and the data transformations necessary and conducted in the process_data.py file
2.  model/Model_Exploratory_Development.ipynb: jupyter notebook that walks through the steps for initial machine learning model refinement

__Output Files__
1.  data/clean_df.csv: flat file resulting from feature_engineering() function in build_model.py file and used for data visualization in app
2.  data/uplift_df.csv: flat file resulting from calc_uplift() function in build_model.py file that captures uplift scores
3.  data/cum_gains_df.csv: flat file containing data for cumulative gains chart


###  Installation
This project utilizes default packages within the Anaconda distribution of Python for the majority of the analysis. In addition, yellowbrick is included for exploratory data visualization for machine learning model development.

A requirements.txt file is included for packages where I had issues without installing upgrades up to at the very least that version of the package.
