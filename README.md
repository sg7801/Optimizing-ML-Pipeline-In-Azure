# Optimizing-ML-Pipeline-In-Azure
# Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

# Summary
This project is based on the Bank Marketing Dataset of a Banking Institution, which contains all information of previous marketing campaigns. Hence, with the help of Azure ML,we seek to predict if a customer will subscribe to the Term Deposit or not by comparing the accuracy of Hyperdrive and Auto ML .

The Best Model of Hyperdrive Run had accuracy of **0.9142407** and that of Auto ML Run had accuracy of **0.91752**. Out of the two methods we concluded that the best performing was **VotingEnsemble Algorithm with 100% Sampling** by AutoML Run.

# Scikit-learn Pipeline

**1. Pipeline architecture:**

I started with the Training Script - train.py which used the Scikit-Learn Logistic Regression. It starts with a clean_data function that cleans the missing values from the dataset and one hot encodes data. I passed the required parameters and then imported the data from the specific URL using **TabularDatasetFactory**. Then, the data was split into the train and test sets. Finally, parameters were passed in the **Logistic Regression Algorithm**.

Further, in the udacity-project.ipynb file, I specified the Parameter Sampler - **RandomParameterSampling** that used the discrete data for hyperparameters. Then I specified the **BanditPolicy** for early termination. Finally after uploading the data to datastore I ran the Hyperdrive using Accuracy as the primary metric.
The BanditPolicy was chosen as the Early Stopping Policy with **slack_factor = 0.02 and evaluation_interval = 1.**

The Best Model of Hyperdrive had **Accuracy of 0.9142407, Regularization Strength of 0.013738337356206531 and Max iterations of 100.** This resulted in value of **'--C'  0.013738337356206531 and '--max_iter' = 100**.

**2. What are the benefits of the parameter sampler you chose?**

The Parameter Sampler chosen was - RandomParameterSampling. The major edge it has over other Samplers is of choosing random values from the search space with ease. It can choose values for the hyperparameters by exploring wider pool of values than others.

**3. What are the benefits of the early stopping policy you chose?**

In this project we used the BanditPolicy as the early stopping policy with parameters evaluation_interval, slack_factor, slack_amount and delay_evaluation.
The search termination takes place if the run does not fall in slack factor or slack amount of evaluation metric range with respect to best performing run.
In this project, the value of the **slack_factor = 0.02**. Starting at specified evaluation_interval, any run resulting in smaller value of primary metric gets cancelled automatically.

# AutoML

Firstly, We used the **TabularDatasetFactory** to create a dataset from the provided link and then used the **clean_data function** to clean and one-hot encode data. Then we split the train and test sets and upload them to datastore.
Then, we define the task as **Classification** with **accurcay** as primary metric and **5 n-cross validations**. 
After the submission, we found that **VotingEnsemble Algorithm** resulted with the best model with **accuracy 0.91752, precision_score_weighted 0.9142882464020461 and precision_score_micro 0.9175154736784336**. Enabling of the automatic featurisation resulted in **Data guardrails** including Class balancing detection, Missing feature values imputation and High cardinality feature detection that checks over the input data to ensure quality in training the model.

# Pipeline comparison

Clearly,the AutoML with accuracy of 0.91752 outperformed the Hyperdrive run since AutoML checks many algorithms whereas Hyperdrive sticks to one. However, there was not much difference in the accuracy of best models of both of them. They differ in many dimensions like Parameters, Algorithms and Configuration. 

# Future work

The major areas of future improvement involve the running of the model for much longer time and trying different parameters to get even better accuracy. 
We can implement and use various other algorithms other than Logistic Regression which might provide us better results. 
We can try different values of n-cross validation, '--C' and '--max_iter' for better results.

# Proof of cluster clean up
