# Arvato Bertelsmann Customer Segmentation


## 1. Project Motivation

The aim of this project is to analyze the customer segmentation Report for Arvato Financial Services, In order to increase efficiency in the customer acquisition process of a mail-order company.

My project <a href="https://github.com/ebulutbaltaci/Arvato_Bertelsmann_Customer_Segmentation" target="_blank">GitHub</a> repository.

Here is blog post : <a href="https://ebulutbaltaci.medium.com/arvato-bertelsmann-customer-segmentation-7cb6cbecbcc7" target="_blank">Medium</a>

## 2. Instructions

In this project, demographic data and customer data of German customers were analyzed in order to realize Customer Segmentation and Customer Gain.

This project is to help the Mail-Order company get new customers to sell their organic products. The goal of this project is to understand customer demographics compared to the general population to decide whether to approach a person for future products. 


## 3. Overview

### Data Origin
The files used in the project:

- `Udacity_AZDIAS_Subset.csv` : Demographic data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
- `Udacity_CUSTOMERS_Subset.csv` : Demographic data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
- `Data_Dictionary.md` : Information file about the features in the provided datasets.
- `AZDIAS_Feature_Summary.csv` : Summary of feature attributes for demographic data.
- `Identify_Customer_Segments.ipynb` : Jupyter Notebook divided into sections and guidelines for completing the project. The notebook provides more details and tips than the outline given here.


The notebook is divided into below parts:

### 3.0.  Get to Know the Data

The data structure was examined and data preprocessing was done.

### 3.1. Customer Segmentation Report

Unsupervised learning techniques, PCA (Major Component Analysis) and K-mean clustering were used to perform customer segmentation and determine the key customer characteristics of customers.

### 3.2. Supervised Learning Model

Finally, with demographic information for the goals of a marketing campaign for the company, I used the Logistic Regression, Random Forest Classifier, and Gradient Boosting Classifier models to predict which people are most likely to convert to customers.

### 3.4. Kaggle Competition 

Once I've chosen a model, I use it to make predictions on the campaign data as part of a Kaggle Competition.


## 4. Conclusion

This project, provided by Udacity partners at Bertelsmann Arvato Analytics, the real-life demographics data of Germany population and customer segment was analyzed.

The most challenging part of the project was the cleaning of NAN and NULL values. One of the most difficult steps had 366 columns to analyze and not all of them had an explanation. It was necessary to have a thorough understanding of the data in order to eliminate missing values and possible outliers. For this, it was necessary to both understand the data and determine the data to be cleaned and edited well. I paid great attention to these in my transactions.

In the uncontrolled part, dimensional reduction using PCA was performed to 72 latent features defining 80% of the variance explained. Clustering K-means into 6 clusters determined the cluster that should be the company's target customers.

The party that is open to improvement in the project, it is necessary to make more fine tuning to use different models and to get a better score in the data engineering department. It takes a long time and patience because it is necessary not to overlook very fine points. Another approach can be used to handle incomplete and misleading data. These changes can improve the performance of our model.

Finally, the Gradient Boosting Classifier was selected and parameterized to create a controlled model and make predictions on a test dataset on Kaggle. We achieved 79.7% ROC.


## 5. Software Requirements

This project uses **Python 3**.


## 6. Libraries Used

I use Python3. Here are the libraries I used in my Jupyter Notebook:

1. pandas
2. numpy
3. matplotlib.pyplot
4. seaborn
5. lightgbm
6. random
7. sklearn.impute
8. sklearn.preprocessing
9. sklearn.decomposition
10. sklearn.cluster
11. sklearn.pipeline
12. sklearn.model_selection
13. sklearn.metrics
14. scipy
15. math
16. LogisticRegression
17. RandomForestClassifier
18. GradientBoostingClassifier



## 7.Requirements

To start working, the following libraries need to be installed or updated versions:

- `pip install -U lightgbm`
- `pip install -U scikit-learn`
- `pip install -U imbalanced-learn`


## 8. Licensing, Authors, Acknowledgements

This project is apart of Udacity's Data Science Nanodegree Program, which provides initial starter code for the project. Additionally, the original datasets are provided in part by Arvato Bertelsmann.