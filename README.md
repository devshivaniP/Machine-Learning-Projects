# Internship Projects

## Overview

Welcome to the repository for the projects I developed during my internship at CodSoft. This README file provides an overview of the projects, including their purpose, key features, technologies used, and instructions for setup and usage.

## Table of Contents

1. [Movie Rating Prediction](#movie-rating-prediction)
    - [Purpose](#purpose)
    - [Dataset](#dataset)
    - [Key Features](#key-features)
    - [Technologies Used](#technologies-used)
    - [Usage](#usage)
    - [Acknowledgements](#acknowledgements)

2. [Iris Flower Classification](#iris-flower-classification)
    - [Purpose](#purpose-1)
    - [Dataset](#dataset-1)
    - [Key Features](#key-features-1)
    - [Technologies Used](#technologies-used-1)
    - [Usage](#usage-1)
    - [Acknowledgements](#acknowledgements-1)

3. [Credit Card Fraud Detection](#credit-card-fraud-detection)
    - [Purpose](#purpose-2)
    - [Dataset](#dataset-2)
    - [Key Features](#key-features-2)
    - [Technologies Used](#technologies-used-2)
    - [Usage](#usage-2)
    - [Acknowledgements](#acknowledgements-2)


## Movie Rating Prediction

### Purpose

The purpose of this project is to develop a machine learning model that can accurately predict movie ratings. This helps streaming services and movie recommendation systems provide personalized recommendations to users.

## Dataset

The dataset used for this project is the [IMDB Movies India dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies), which contains millions of ratings and tag applications across thousands of movies. This dataset includes user ratings, movie information, and user demographics.

### Key Features

- Data Preprocessing: Handling missing values, encoding categorical features, and splitting data into training and testing sets.
- Feature Engineering: Creating new features to improve model performance.
- Model Training: Training various machine learning models to predict movie ratings.
- Model Evaluation: Evaluating model performance using metrics such as MAPE (Mean Absolute Percentage Error) and accuracy. 
- Predictive Analysis: Using the trained model to predict ratings for new movies or users.

### Technologies Used

- Programming Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Tools: Jupyter Lab, Git

### Usage

#### Training the Model

1. Open the Jupyter Notebook.
2. Open the `IMDB Movie Rating.ipynb` file.
3. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

#### Evaluating the Model

The model evaluation section in the notebook includes various metrics to assess the performance of the rating prediction model. Metrics include accuracy and MAPE(Mean Absolute Percentage Error).

#### Predicting Ratings

To use the trained model for predicting fraud on new movies or users, you can run the prediction section in the notebook.

### Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies) for providing the dataset.

- [Scikit-learn](https://scikit-learn.org/) for machine learning tools and algorithms.

- [CodSoft](https://www.codsoft.in/) for the opportunity to work on this project.


## Iris Flower Classification

### Purpose

The purpose of this project is to build a machine learning model that can accurately classify iris flowers into one of three species based on their sepal and petal measurements. This project serves as a demonstration of basic machine learning techniques and practices.

### Dataset

The dataset used for this project is the famous [Iris Flower Dataset](https://www.kaggle.com/datasets/arshid/iris-flower-dataset). It consists of 150 samples, each with four features: sepal length, sepal width, petal length, and petal width. There are 50 samples for each of the three species.

### Key Features

- Data Preprocessing: Handling missing values, scaling features, and splitting data into training and testing sets.
- Model Training: Training various machine learning models to classify iris species.
- Model Evaluation: Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score.

### Technologies Used

- Programming Languages: Python 
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- Tools: Jupyter Lab, Git

### Usage

#### Training the Model

1. Open the Jupyter Notebook.
2. Open the `Iris Flower Classification.ipynb` file.
3. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

#### Evaluating the Model

The model evaluation section in the notebook includes various metrics to assess the performance of the flower classification model. Metrics include accuracy, precision, recall, and F1-score.

#### Predicting Species

To use the trained model for predicting fraud on new iris samples, you can run the prediction section in the notebook.

### Acknowledgements

- [Kaggle](https://www.kaggle.com/datasets/arshid/iris-flower-dataset) for providing the dataset.

- [Scikit-learn](https://scikit-learn.org/) for machine learning tools and algorithms.

- [CodSoft](https://www.codsoft.in/) for the opportunity to work on this project.



## Credit Card Fraud Detection

### Purpose

The purpose of this project is to develop a machine learning model that can accurately detect fraudulent credit card transactions. This helps financial institutions prevent fraud and protect customers from unauthorized transactions.

### Dataset

The dataset used for this project is the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by credit cards in September 2013 by European cardholders. The dataset has 284,807 transactions, including 492 fraudulent transactions.

### Key Features

- Data Preprocessing: Handling missing values, scaling features, and dealing with imbalanced data.
- Feature Engineering: Creating new features to improve model performance.
- Model Training: Training various machine learning models to identify fraud.
- Model Evaluation: Evaluating model performance using metrics such as accuracy, precision, recall, and F1-score.

### Technologies Used

- Programming Languages: Python
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn, pycaret
- Tools: Jupyter Lab, Git


### Usage

#### Training the Model

1. Open the Jupyter Notebook.
2. Open the `Credit Card Fraud Detection.ipynb` file.
3. Run the notebook cells to preprocess the data, train the model, and evaluate its performance.

#### Evaluating the Model

The model evaluation section in the notebook includes various metrics to assess the performance of the fraud detection model. Metrics include accuracy, precision, recall, F1-score, and ROC-AUC score.

#### Predicting Fraud

To use the trained model for predicting fraud on new transactions, you can run the prediction section in the notebook.

### Acknowledgements

- [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) for providing the dataset.
- [Imbalanced-learn](https://imbalanced-learn.org/stable/) for resampling techniques.
- [Scikit-learn](https://scikit-learn.org/) for machine learning tools and algorithms.
- [Pycaret](https://pycaret.org/) for training data on various models simultaneously.

- [CodSoft](https://www.codsoft.in/) for the opportunity to work on this project.

## Setup Instructions

### Prerequisites

- Python 3.6 or higher
- Jupyter Notebook

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


