# Lead Scoring Prediction

## Table of Contents
1. [Background Overview](#background-overview)
2. [Data Dictionary](#data-dictionary)
3. [Executive Summary](#executive-summary)
4. [Methodology](#methodology)
   1. [Data Preprocessing](#1-data-preprocessing)
   2. [Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
   3. [Machine Learning Models](#4-machine-learning-models)
      - [Modeling Approaches](#modeling-approaches)
      - [Model Performance](#model-performance)
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
   - **Correlation Matrix**: A heatmap was generated to visualize the correlation matrix of the features in the dataset. This matrix indicated which variables had strong positive or negative correlations with the target variable `BUY` and among themselves. 

   - **Interaction Heatmaps**: Heatmaps were also created to visualize interactions between features, specifically focusing on the effects of images, reviews, specifications, and FAQs on purchasing decisions. The following insights were derived from this analysis:

     - **IMAGES vs. REVIEWS**:
       - **IMAGES = 0, REVIEWS = 0**: 90.48% did not purchase, while 9.52% did.
       - **IMAGES = 0, REVIEWS = 1**: 46.43% did not buy, while 53.57% made a purchase. This suggests that viewing reviews significantly increases the chances of making a purchase even without images.
       - **IMAGES = 1, REVIEWS = 0**: 77.78% did not buy, while 22.22% made a purchase. Here, viewing images alone increases the probability of purchase compared to viewing nothing, but less than when reviews are viewed.
       - **IMAGES = 1, REVIEWS = 1**: 41.67% did not buy, while 58.33% made a purchase. This combination has the highest probability of leading to a purchase, highlighting the strong interaction effect between viewing images and reviews.

     - **SPECS vs. REVIEWS**:
       - **SPECS = 0, REVIEWS = 0**: 89.29% did not purchase, while 10.71% did. Users lacking exposure to product information are unlikely to buy.
       - **SPECS = 0, REVIEWS = 1**: 33.33% did not purchase, while 66.67% made a purchase. Reviews have a strong positive influence on purchasing decisions even when specifications are missing.
       - **SPECS = 1, REVIEWS = 0**: 75% did not buy, while 25% did. Having specifications increases likelihood of purchase compared to having neither, but it is significantly lower when reviews are included.
       - **SPECS = 1, REVIEWS = 1**: 53.57% did not buy, while 46.43% did. Notable improvement in purchase likelihood when both pieces of information are available.

     - **IMAGES vs. FAQ**:
       - **IMAGES = 0, FAQ = 0**: 53.85% did not purchase, while 46.15% did. A relatively balanced outcome where almost half of users still made purchases, despite lacking product visuals and FAQs.
       - **IMAGES = 0, FAQ = 1**: 78.26% did not purchase, while only 21.74% made a purchase. Viewing the FAQ alone is not effective in driving purchases if users do not see images.
       - **IMAGES = 1, FAQ = 0**: 63.33% did not buy, while 36.67% did. Having images available slightly increases the likelihood of purchase compared to having neither images nor FAQ.
       - **IMAGES = 1, FAQ = 1**: 57.14% did not make a purchase, while 42.86% did. Combining these two types of product information positively affects purchase likelihood.

     - **WARRANTY vs. SPECS**:
       - **WARRANTY = 0, SPECS = 0**: 65.12% did not purchase, while 34.88% did. Users lacking both types of information still make purchases, but the conversion rate is low.
       - **WARRANTY = 0, SPECS = 1**: 80.95% did not purchase, while only 19.05% made a purchase. Specifications alone are not effective in driving purchases without warranty information.
       - **WARRANTY = 1, SPECS = 0**: 61.83% did not purchase, while 38.17% did. Having warranty information increases likelihood of purchase compared to having neither.
       - **WARRANTY = 1, SPECS = 1**: 48.15% did not make a purchase, while 51.85% did. Providing both types of information significantly increases the likelihood of purchase.

     - **COMPARE_SIMILAR vs. SPONSORED_LINKS**:
       - **COMPARE_SIMILAR = 0, SPONSORED_LINKS = 0**: 71.43% did not purchase, while 28.57% did. Less likely to make a purchase when not engaging with either feature.
       - **COMPARE_SIMILAR = 0, SPONSORED_LINKS = 1**: 76.19% did not purchase, while 23.81% made a purchase. Clicking on sponsored links without comparing similar items leads to a lower conversion rate.
       - **COMPARE_SIMILAR = 1, SPONSORED_LINKS = 0**: 66.67% did not make a purchase, while 33.33% did. Comparing similar items positively impacts purchase likelihood compared to not engaging with both features.
       - **COMPARE_SIMILAR = 1, SPONSORED_LINKS = 1**: 47.06% did not make a purchase, while 52.94% did. Combining both features significantly increases the likelihood of making a purchase, with more than half of users converting.


### **Key Insights from EDA**:
   - Users who viewed reviews significantly increased their likelihood of making a purchase compared to those who did not.
   - The analysis of the correlation matrix revealed that features like `REVIEWS` and `IMAGES` had strong positive correlations with `BUY`, highlighting their importance in the purchasing decision process.


### 3. **Machine Learning Models**:

   #### **Model Performance**:
   - **Evaluation Metrics**: 
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - ROC-AUC score

   #### **Results**:
   - **Best Performing Model**:


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
