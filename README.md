# Lead Scoring Prediction

## Table of Contents
1. [Background Overview](#background-overview)
2. [Data Dictionary](#data-dictionary)
3. [Executive Summary](#executive-summary)
4. [Methodology](#methodology)
   1. [Data Preprocessing](#1-data-preprocessing)
   2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   3. [Feature Engineering](#3-feature-engineering)
   4. [Machine Learning Models](#4-machine-learning-models)
      - [Modeling Approaches](#modeling-approaches)
      - [Model Performance](#model-performance)
   5. [Interaction Analysis](#5-interaction-analysis)
5. [Recommendations](#recommendations)
6. [Further Studies and Limitations](#further-studies-and-limitations)

## Background Overview

This project investigates the browsing behavior of users and how various actions they take on a website influence their purchasing decisions. In an increasingly competitive digital marketplace, understanding what drives users to make purchases is critical for businesses to optimize their websites and increase conversion rates. The primary objective of this project is to identify key interactions between different browsing behaviors (such as viewing images, reviews, or product specifications) and their effect on purchase likelihood. The results provide actionable insights to help improve website user experience and drive higher sales.

## Data Dictionary

The dataset used in this project contains information about user sessions on a website and includes the following variables:

- **SESSION_ID**: Unique identifier for each user session.
- **IMAGES**: Indicates whether the user viewed images during the session (1 = Yes, 0 = No).
- **REVIEWS**: Indicates whether the user looked at product reviews (1 = Yes, 0 = No).
- **FAQ**: Indicates whether the user viewed the FAQ section (1 = Yes, 0 = No).
- **SPECS**: Indicates whether the user viewed product specifications (1 = Yes, 0 = No).
- **SHIPPING**: Indicates whether the user viewed shipping information (1 = Yes, 0 = No).
- **BRO_TOGETHER**: Indicates whether the user browsed similar items together (1 = Yes, 0 = No).
- **COMPARE_SIMILAR**: Indicates whether the user compared similar items (1 = Yes, 0 = No).
- **VIEW_SIMILAR**: Indicates whether the user viewed similar items (1 = Yes, 0 = No).
- **WARRANTY**: Indicates whether the user checked product warranty (1 = Yes, 0 = No).
- **SPONSORED_LINKS**: Indicates whether the user clicked on sponsored links (1 = Yes, 0 = No).
- **BUY**: Indicates whether the user made a purchase during the session (1 = Yes, 0 = No).

## Executive Summary

This project explores how different user interactions on a product webpage influence the likelihood of a purchase. By analyzing user browsing behavior—specifically, viewing images, reading reviews, and checking product specifications—we can identify which combinations of actions are most likely to lead to a purchase. 

**Key Findings**:
- **Reviews drive purchases**: Users who viewed reviews were significantly more likely to make a purchase, even without viewing images.
- **Synergy between images and reviews**: Users who viewed both images and reviews had the highest likelihood of making a purchase.
- **Specifications play a supportive role**: Product specifications alone slightly increase purchase probability, but when combined with reviews, the likelihood improves.
- **Actionable Insights**: To optimize conversions, websites should ensure that product reviews and images are readily available together, and consider emphasizing these elements in the user experience.

## Methodology

The analysis followed a structured approach:

### 1. **Data Preprocessing**:
   - The dataset was cleaned.
   - Features were standardized and categorical variables were encoded where necessary to prepare the data for machine learning modeling.

### 2. **Exploratory Data Analysis (EDA)**

EDA is an essential step in understanding the dataset's structure and relationships before applying machine learning models. In this project, the EDA process helped uncover key patterns and correlations between user actions and purchase decisions. The following techniques were applied during the EDA phase:

#### **2.1. Descriptive Statistics**:
   - **Summary Statistics**: Summary statistics were generated for the dataset, including mean, median, mode, variance, and standard deviation for numerical variables like `IMAGES`, `REVIEWS`, and `BUY`.
   - **Class Distribution**: The target variable `BUY` was analyzed to determine its distribution:
     ```
     BUY
     0    315
     1    185
     ```
     This indicates an imbalance in the dataset, with significantly more instances of non-purchases (0) compared to purchases (1). This class imbalance could impact model performance and should be addressed during model training.

#### **2.2. Contingency Tables**:
   - **Crosstab Analysis**: Contingency tables were created to analyze the relationship between the categorical variables, such as `REVIEWS` and `BUY`. This analysis examined how many users who viewed reviews went on to make a purchase compared to those who did not. The results provided insights into the impact of reviewing behavior on purchasing decisions.

#### **2.3. Heatmaps**:
   - **Correlation Matrix**: A heatmap was generated to visualize the correlation matrix of the features in the dataset. This matrix indicated which variables had strong positive or negative correlations with the target variable `BUY` and among themselves. For example, a strong correlation between `REVIEWS` and `BUY` would suggest that users who read reviews are more likely to make a purchase.
   - **Interaction Heatmaps**: Heatmaps were also created to visualize interactions between features, such as `IMAGES` vs. `REVIEWS`, and their combined effect on `BUY`. These visualizations helped to identify potential synergies between user actions.

### **Key Insights from EDA**:
   - Users who viewed reviews significantly increased their likelihood of making a purchase compared to those who did not.
   - The analysis of the correlation matrix revealed that features like `REVIEWS` and `IMAGES` had strong positive correlations with `BUY`, highlighting their importance in the purchasing decision process.

### 3. **Feature Engineering**:
   - New interaction features (e.g., `IMAGES_AND_REVIEWS`, `SPECS_AND_REVIEWS`) were created to capture the combined effects of multiple user actions.
   - These new features were analyzed to see how they influenced purchase behavior.

### 4. **Machine Learning Models**:

   Several machine learning models were applied to predict whether a user would make a purchase based on their browsing behavior:

   #### **Modeling Approaches**:
   - **Logistic Regression**: This baseline model was used to estimate the probability of purchase based on user interactions with the website.
   - **Random Forest Classifier**: This model was used to capture more complex interactions between user behaviors and purchases. The Random Forest algorithm creates an ensemble of decision trees to improve predictive performance.
   - **Gradient Boosting Classifier**: This model was applied to further improve accuracy by iteratively correcting the errors made by previous trees.
   - **XGBoost**: This boosted tree algorithm was used for its speed and performance, offering better handling of complex patterns in the dataset.

   #### **Model Performance**:
   - **Evaluation Metrics**: 
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC score
   - **Cross-validation**: 5-fold cross-validation was applied to ensure that the models generalize well to unseen data.
   - **Model Tuning**: Hyperparameter tuning using GridSearchCV was performed for Random Forest, Gradient Boosting, and XGBoost to find the optimal parameters.
   
   #### **Results**:
   - **Best Performing Model**: XGBoost yielded the highest performance with an accuracy of 86%, a precision of 0.83, and an ROC-AUC score of 0.89.
   - **Feature Importance**: 
     - The most important features were `REVIEWS`, `IMAGES`, and `SPECS`, with interaction features like `IMAGES_AND_REVIEWS` contributing significantly to the purchase prediction.
     - Users who viewed reviews, images, and product specifications were much more likely to make purchases.

### 5. **Interaction Analysis**:
   - Cross-tabulation was used to examine how pairs of features (e.g., viewing images and reviews) interact and affect purchase decisions.
   - Heatmaps were generated to visualize these interactions and uncover patterns.

## Recommendations

Based on the analysis, the following actionable recommendations are suggested:

- **Highlight Product Reviews and Images Together**: To maximize conversion rates, ensure that product reviews are easily accessible and prominently displayed alongside product images. 
- **Engage Users with Additional Content**: Besides images and reviews, consider offering customer testimonials, videos, or product comparison charts to further influence purchasing decisions.
- **Drive Focus on Reviews**: Since reviews have a significant impact on purchase behavior, businesses should encourage users to leave detailed reviews and display them effectively on product pages.

## Further Studies and Limitations

This analysis provides valuable insights, but there are limitations and opportunities for further research:

1. **Broader User Data**: The dataset only includes information about a limited set of user interactions. Additional factors, such as pricing, promotions, and competitor products, could provide a more comprehensive understanding of purchasing behavior.
   
2. **Temporal Analysis**: Future studies could explore how user behavior and purchase patterns change over time (e.g., during sales or holiday seasons).

3. **Personalization**: Different users may respond to different types of information. Future research could explore how user segmentation (e.g., based on demographics) influences what drives their purchasing decisions.
