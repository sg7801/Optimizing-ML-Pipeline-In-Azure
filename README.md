# Optimizing-ML-Pipeline-In-Azure
# Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

# Summary
This project is based on the Bank Marketing Dataset of a Portugese Banking Institution, which contains all information of previous marketing campaigns. Hence, with the help of Azure ML,we compare the accuracy of Hyperdrive and Auto ML to predict if a customer will subscribe to the Term Deposit or not.

The Best Model of Hyperdrive Run had accuracy of -- and that of Auto ML Run had accuracy of ---. On comparing the results of the two methods we concluded that the best performing out of the two was ---.

# Scikit-learn Pipeline

1. Pipeline architecture, including data, hyperparameter tuning, and classification algorithm:

I started with the Training Script called train.py which used the Scikit-Learn Logistic Regression. It starts with a clean_data function that cleans the missing values from the dataset and one hot encodes data. I passed the required parameters and then imported the data from the specific URL using TabularDatasetFactory. Then, I split the data into the train and test sets. 

Further, in the udacity-project.ipynb file, I specified the Parameter Sampler - RandomParameterSampling that used the discrete data for hyperparameters. Then I specified the BanditPolicy for early termination. Finally after uploading the data to datastore I ran the Hyperdrive using Accuracy as the primary metric.

2. What are the benefits of the parameter sampler you chose?

The Parameter Sampler we chose was - RandomParameterSampling. It randomly selects the hyperparameters in the search space from the discrete set of values or from a continuous range. In this project for the hyperparamters C and max_iter, this will choose values randomly from a wider pool of values in easier way as compared to other parameter samplers.

3. What are the benefits of the early stopping policy you chose?

In this project we used the BanditPolicy as the early stopping policy. This takes the parameters such as evaluation_interval, slack_factor, slack_amount and delay_evaluation.
It terminates the search if the run does not fall in range of slack factor or slack amount of the evaluation metric with respect to the best performing run. In this project I took the value of the slack_factor = 0.02.
Hence, If the best Performance metric for a run is A, it will compare (A + (A * 0.02)) to primary metric of all runs starting at specified evaluation_interval. If any of them results in smaller value then the run gets cancelled.

# AutoML

Model : 

Hyperparameters :

# Pipeline comparison

# Future work

The major areas of future improvement involve the running of the model for much longer time to get even better accuracy. Further, We can implement and use various other algorithms other than Logistic Regression and different parameters in AutoML which might provide us better results. Along with that, we can leave the task of cleaning data on the run itself rather than specifically creating a function clean_data in train.py in the project. 

# Proof of cluster clean up
