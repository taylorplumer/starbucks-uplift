# starbucks-uplift
Build an uplift model with machine learning pipeline to prioritize promotional outreach

### Summary
This project contains simulated Starbucks data of promotional offers and purchasing behavior. 

The deliverable is a dashboard that contains visualizations, most notably the cumulative gains chart, which aid busines users in their assessment of the uplift model performance and related target segment characteristics. More details regarding the deliverable are below in the Results section.

The repository contains working code for running an ETL pipeline, ML pipeline, and Flask app locally. Instructions are below.

### Problem Statement

An uplift model seeks to predict the incremental value from sending a promotion.

The objective is to create an uplift model that scores and prioritizes events where offer is both viewed and a transaction is completed within the offer duration.

A classification model will be developed to predict the results of an offer sent, which is termed as an "event" throughout this documentation.

The classifications will be four target segments, which can be visualized in a 2x2 matrix

![](https://www.predictiveanalyticsworld.com/patimes/wp-content/uploads/2017/03/Mike-Thurber-Graphic-2.png)

The model will be evaluated with the use of a cumulative gains chart. 


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores processed data in a csv 
        `python data/process_data.py data/processed_data.py`
    - To run ML pipeline that trains classifier and outputs various csv files in data folder
        `python model/build_model.py data/dataframe.csv`

2. Run the following command in the app's directory to run your web app.
    `python myapp.py`

3. Go to http://0.0.0.0:3001/


### Important Files:

1.  data/process_data.py: ETL script to clean and saves data to csv file
2.  models/train_classifier.py: builds, trains, evaluates, and  outputs results of machine learning classifier prediction
4.  app/run.py: this file is used to run the Flask application


###  Installation
This project utilizes default packages within the Anaconda distribution of Python for the majority of the analysis. In addition, yellowbrick is included for exploratory data visualization for machine learning model development.

A requirements.txt file is included for packages where I had issues without installing upgrades up to at the very least that version of the package.
