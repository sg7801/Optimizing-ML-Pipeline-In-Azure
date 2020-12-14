# Optimizing ML Pipeline In Azure
# Overview
This project is part of the Udacity Azure ML Nanodegree. In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model. This model is then compared to an Azure AutoML run.

# Summary
This project is based on the Bank Marketing Dataset of a Banking Institution, which contains all information of previous marketing campaigns. Hence, with the help of Azure ML we seek to predict if a customer will subscribe to the Term Deposit or not by comparing the accuracy of Hyperdrive and Auto ML .

The Best Model of Hyperdrive Run had accuracy of **0.9142407** and that of Auto ML Run had accuracy of **0.91752**. Out of the two methods we concluded that the best performing was **VotingEnsemble Algorithm with 100% Sampling** by AutoML Run.

# Scikit-learn Pipeline

**1. Pipeline architecture:**

I started with the Training Script - train.py which used the Scikit-Learn Logistic Regression. It starts with a clean_data function that cleans the missing values from the dataset and one hot encodes data. I passed the required parameters and then imported the data from the specific URL using **TabularDatasetFactory**. Then, the data was split into the train and test sets. Finally, parameters were passed in the **Logistic Regression Algorithm**.

Further, in the udacity-project.ipynb file, I specified the Parameter Sampler - **RandomParameterSampling** that used the discrete data for hyperparameters. Then I specified the **BanditPolicy** for early termination. Finally after uploading the data to datastore I ran the Hyperdrive using Accuracy as the primary metric.
The BanditPolicy was chosen as the Early Stopping Policy with **slack_factor = 0.02 and evaluation_interval = 1.**

The Best Model of Hyperdrive had **Accuracy of 0.9142407, Regularization Strength of 0.013738337356206531 and Max iterations of 100.** This resulted in value of **'--C' = 0.013738337356206531 and '--max_iter' = 100**.

**2. What are the benefits of the parameter sampler you chose?**

The Parameter Sampler chosen was - RandomParameterSampling. The major edge it has over other Samplers is of choosing random values from the search space with ease. It can choose values for the hyperparameters by exploring wider pool of values than others.

**3. What are the benefits of the early stopping policy you chose?**

In this project we used the BanditPolicy as the early stopping policy with parameters evaluation_interval, slack_factor, slack_amount and delay_evaluation.
The search termination takes place if the run does not fall in slack factor or slack amount of evaluation metric range with respect to best performing run.
We took the value of the **slack_factor = 0.02, evaluation_interval = 1, delay_evaluation = 0 and slack_amount = none.**. Starting at specified evaluation_interval, any run resulting in smaller value of primary metric gets cancelled automatically.

**Runs of Hyperdrive**

![Hyperdrive Runs](https://user-images.githubusercontent.com/61888364/99191995-559f1a00-2796-11eb-949d-fe152c2c5985.png)

**Visualisation of Accuracy**

![Hyperdrive Run](https://user-images.githubusercontent.com/61888364/99191951-1670c900-2796-11eb-9675-b944bbfd16f1.png)

# AutoML

Firstly, We used the **TabularDatasetFactory** to create a dataset from the provided link and then used the **clean_data function** to clean and one-hot encode data. Then we split the train and test sets and upload them to datastore.
Then, we define the task as **Classification** with **accurcay** as primary metric and **5 n-cross validations**. 
After the submission, we found that **VotingEnsemble Algorithm** resulted with the best model with **accuracy 0.91752, precision_score_weighted 0.9142882464020461 and precision_score_micro 0.9175154736784336**. Enabling of the automatic featurisation resulted in **Data guardrails** including Class balancing detection, Missing feature values imputation and High cardinality feature detection that checks over the input data to ensure quality in training the model.

**Runs**

![AutoML Runs](https://user-images.githubusercontent.com/61888364/99192039-a31b8700-2796-11eb-8b06-f23bceda752e.png)

**Metrics**

![AutoML Metrics](https://user-images.githubusercontent.com/61888364/99192071-d3fbbc00-2796-11eb-847f-72caefd64930.png)

**Explainations(preview)**

![AutoML Explaination](https://user-images.githubusercontent.com/61888364/99192099-00afd380-2797-11eb-96c0-91601f0b1412.png)

![AutoML Explaination 2](https://user-images.githubusercontent.com/61888364/99192106-0e655900-2797-11eb-998a-5cca39e140eb.png)

# Pipeline comparison

Clearly,the AutoML with accuracy of 0.91752 outperformed the Hyperdrive run since AutoML checks many algorithms whereas Hyperdrive sticks to one. However, there was not much difference in the accuracy of best models of both of them. They differ in many dimensions like Parameters, Algorithms and Configuration. 

# Future work

The major areas of future improvement involve the running of the model for much longer time and trying different parameters to get even better accuracy. 
We can implement and use various algorithms other than Logistic Regression which might provide us better results. 
We can try different values of n-cross validation, '--C' and '--max_iter'.

# Proof of cluster clean up

**Images of cluster marked for deletion:**

![Proof](https://user-images.githubusercontent.com/61888364/99192151-48365f80-2797-11eb-9c16-183421167348.png)

![Proof2](https://user-images.githubusercontent.com/61888364/99192155-54bab800-2797-11eb-8bb9-270c39576fd9.png)
