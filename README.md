# Predicting Purchase Intent Using User Session Data

## Table of Contents
1. [Background Overview](#background-overview)
2. [Data Dictionary](#data-dictionary)
3. [Executive Summary](#executive-summary)
4. [Insights Deep Dive](#insights-deep-dive)
5. [Recommendations](#recommendations)
6. [Further Studies and Limitations](#further-studies-and-limitations)

---

## Background Overview
This project aims to predict whether a user will make a purchase based on various features collected during their online shopping session. By understanding key behaviors such as reviewing product specs, checking warranties, or clicking on sponsored links, we can better predict purchasing intent and provide recommendations for improving e-commerce strategies.

---

## Data Dictionary
The dataset contains the following columns:

- **SESSION_ID**: Unique identifier for each session.
- **IMAGES**: Whether the user viewed images (1 for yes, 0 for no).
- **REVIEWS**: Whether the user looked at reviews (1 for yes, 0 for no).
- **FAQ**: Whether the user viewed the FAQ section (1 for yes, 0 for no).
- **SPECS**: Whether the user viewed the product specifications (1 for yes, 0 for no).
- **SHIPPING**: Whether the user viewed shipping information (1 for yes, 0 for no).
- **BRO_TOGETHER**: Whether the user browsed similar items together (1 for yes, 0 for no).
- **COMPARE_SIMILAR**: Whether the user compared similar items (1 for yes, 0 for no).
- **VIEW_SIMILAR**: Whether the user viewed similar items (1 for yes, 0 for no).
- **WARRANTY**: Whether the user checked the warranty (1 for yes, 0 for no).
- **SPONSORED_LINKS**: Whether the user clicked on sponsored links (1 for yes, 0 for no).
- **BUY**: Whether the user made a purchase (1 for yes, 0 for no).

### Target Variable
The target column **BUY** indicates whether a user completed a purchase during the session. The goal is to use the provided features to predict the likelihood of a purchase.

---

## Executive Summary
By analyzing user behavior on the platform, we built a predictive model that can identify potential buyers based on their interactions (e.g., viewing reviews, comparing similar items). We evaluated several models, including Gaussian Naive Bayes, Logistic Regression, and Decision Trees, selecting the best-performing one to predict future purchases.

---

## Insights Deep Dive
### Data Preprocessing Steps
1. **Data Types Check**: Verified correct data types for all features.
2. **Descriptive Summary**: Performed a summary to understand the data distribution.
3. **Correlation Analysis**: Investigated how each feature correlates with the target variable, **BUY**. Key insights include:
   - **REVIEWS** has a strong positive correlation with purchasing intent (0.4046).
   - **VIEW_SIMILAR** shows a strong negative correlation (-0.961), indicating users who spend time comparing items are less likely to buy.
   - **BRO_TOGETHER** and **SPONSORED_LINKS** are moderately correlated.

### Target Variable Distribution
The target variable **BUY** has the following distribution:
- Class 0 (No Purchase): 315 sessions
- Class 1 (Purchase): 185 sessions

### Feature Selection
We selected features based on correlation and domain knowledge:
- **REVIEWS**
- **BRO_TOGETHER**
- **COMPARE_SIMILAR**
- **WARRANTY**
- **SPONSORED_LINKS**

### Data Splitting
The data was split into training (70%) and testing (30%) sets to evaluate the model's performance. The goal was to ensure that the model generalizes well and does not overfit the training data.

---

## Model Evaluation

### Gaussian Naive Bayes
- **Precision**: 0.79 (Class 0), 0.56 (Class 1)
- **Recall**: 0.67 (Class 0), 0.70 (Class 1)
- **Accuracy**: 68%
- **ROC AUC**: 0.78

### Logistic Regression
- **Precision**: 0.85 (Class 0), 0.62 (Class 1)
- **Recall**: 0.71 (Class 0), 0.79 (Class 1)
- **Accuracy**: 74%
- **ROC AUC**: 0.78

### Decision Tree (Best Model)
- **Precision**: 0.93 (Class 0), 0.64 (Class 1)
- **Recall**: 0.69 (Class 0), 0.91 (Class 1)
- **Accuracy**: 77%
- **ROC AUC**: 0.86

#### Metrics Explained
- **Precision**: Measures how many predicted positive cases (purchases) are actually positive. A higher precision means fewer false positives.
- **Recall**: Measures how many actual positive cases (purchases) were correctly identified. A higher recall means fewer false negatives.
- **Accuracy**: Overall model performance, showing how well the model predicts both positive and negative cases.
- **ROC AUC Score**: A metric showing how well the model distinguishes between the two classes (buyers vs. non-buyers). The closer the score is to 1, the better.

---

## Recommendations
- **Improve User Experience on Product Reviews**: Since users viewing reviews are more likely to make a purchase, it is recommended to enhance the review section to guide users towards a decision.
- **Target Users Browsing Similar Products**: Users comparing or browsing similar items are less likely to buy. Offering personalized recommendations or special deals could increase their chances of making a purchase.
- **Promote Warranty and Sponsored Links**: Users checking warranties or clicking sponsored links show higher purchase intent. Highlight these features during the user journey.

---

## Further Studies and Limitations
- **Class Imbalance**: The dataset shows an imbalance between buyers and non-buyers, which was handled using SMOTE. Further exploration into handling imbalances may provide better insights.
- **Feature Expansion**: Including additional behavioral features, such as time spent on the site, could improve predictive accuracy.
- **Limitations of Decision Trees**: Although Decision Trees performed well, they can sometimes overfit. Further tuning or trying advanced models like Random Forest or XGBoost could improve results.

---

## Conclusion
This project successfully developed a predictive model that can identify potential buyers based on their session activity. By focusing on specific user behaviors and optimizing the purchase journey, businesses can potentially increase conversion rates and improve customer satisfaction.
