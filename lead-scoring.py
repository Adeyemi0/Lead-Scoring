import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

# Page title
st.title("Lead Scoring Prediction App")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Introduction", "Data Processing", "Model Prediction"])

if page == "Introduction":
    st.header("Introduction")
    st.markdown("""
    Welcome to the Lead Scoring Prediction App. This application is designed to help predict the likelihood of a customer making a purchase based on their browsing behavior. The project involves data processing, model training, and making predictions using the trained model.

    ### Data Dictionary
    The dataset used in this project includes the following columns:

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

    The goal is to use these features to predict the `BUY` column, indicating whether a user will make a purchase or not.
    """)

elif page == "Data Processing":
    # Section: Data Processing
    st.header("Data Processing")
    st.markdown("""
    This section covers the data loading, preprocessing, and model training steps.
    """)

    # Step 1: Load the Dataset
    st.subheader("Step 1: Load the Dataset")
    data_path = "web-browsing-data.csv"
    data = pd.read_csv(data_path)
    st.write("#### Dataset Preview")
    st.dataframe(data.head())

    # Step 2: Display Data Types
    st.subheader("Step 2: Display Data Types")
    st.markdown("""
    Understanding data types helps us identify how each column is represented. This is crucial for selecting appropriate preprocessing methods.
    """)
    st.write(data.dtypes)

    # Step 3: Display Data Information
    st.subheader("Step 3: Display Data Information")
    st.markdown("""
    This provides a summary of the dataset, including the number of non-null entries, which helps identify any missing data.
    """)
    buffer = pd.DataFrame(data.describe(include='all')).transpose()
    st.write(buffer)

    # Step 4: Correlation Analysis
    st.subheader("Step 4: Correlation Analysis")
    st.markdown("""
    I computed the correlation between each feature and the target variable `BUY`. This helps identify which features have a significant relationship with the target.
    """)
    st.write(data.corr()['BUY'])

    # Step 5: Target Variable Distribution
    st.subheader("Step 5: Target Variable Distribution")
    st.markdown("""
    I checked the distribution of the target variable `BUY` to understand the balance of classes. This helps determine if we need to handle class imbalance.
    """)
    st.write(data['BUY'].value_counts())

    # Step 6: Feature Selection and Data Splitting
    st.subheader("Step 6: Feature Selection and Data Splitting")
    st.markdown("""
    ### Method of Feature Selection

    We have selected features based on domain knowledge and initial correlation analysis with the target variable (BUY). The chosen features are believed to significantly influence the likelihood of a user making a purchase. The selected features are:
    - **REVIEWS**: Indicates if the user looked at reviews, which could suggest a higher intent to buy.
    - **BRO_TOGETHER**: Indicates if the user browsed similar items together, which may show comparison shopping.
    - **COMPARE_SIMILAR**: Indicates if the user compared similar items, suggesting an active interest in finding the best option.
    - **WARRANTY**: Indicates if the user checked the warranty, implying consideration for purchase.
    - **SPONSORED_LINKS**: Indicates if the user clicked on sponsored links, which could mean they are exploring buying options.

    After selecting these features, we proceed to split the data into training and test sets to evaluate the model's performance.

    #### Selected Features:
    """)
    features = ['REVIEWS', 'BRO_TOGETHER', 'COMPARE_SIMILAR', 'WARRANTY', 'SPONSORED_LINKS']
    st.write(features)

    st.markdown("""
    ### Why Split the Data into Training and Test Sets?

    Splitting the data into training and test sets is a crucial step in machine learning. Here's why:

    1. **Model Training and Evaluation**:
       - **Training Set**: The model is trained using this portion of the data. It learns patterns and relationships between the features and the target variable.
       - **Test Set**: This is used to evaluate the model's performance. The test set acts as unseen data, helping us assess how well the model generalizes to new data.

    2. **Avoid Overfitting**:
       - Overfitting occurs when the model learns the training data too well, including noise and outliers, which negatively affects its performance on new data.
       - By evaluating the model on a separate test set, we can check for overfitting and ensure the model has learned to generalize rather than memorize.

    3. **Reliable Performance Metrics**:
       - Using a test set helps provide a more accurate measure of the model's performance, as it simulates how the model will perform in real-world scenarios where new data is encountered.

    ### Data Splitting:
    - **Test Size**: We are using 30% of the data for testing and 70% for training. This split ensures that a substantial portion of the data is available for training while still providing enough data to reliably test the model's performance.
    - **Random State**: We set a random state to ensure the split is reproducible, meaning every time we run the code, the data is split the same way. This helps in achieving consistent results.
    """)

    X = data[features]
    y = data['BUY']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    st.markdown("""
    **Training Set Size**: {train_size} samples  
    **Test Set Size**: {test_size} samples
    """.format(train_size=len(X_train), test_size=len(X_test)))

    # Step 7: Data Resampling
    st.subheader("Step 7: Data Resampling")
    st.markdown("""
    Using SMOTE (Synthetic Minority Over-sampling Technique), we balance the dataset to handle class imbalance, which can improve the model's ability to learn minority class patterns.
    """)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    st.write("Data balanced using SMOTE")

    # Step 8: Feature Scaling
    st.subheader("Step 8: Feature Scaling")
    st.markdown("""
    I scaled the features to standardize the dataset, which is crucial for certain algorithms (e.g., Logistic Regression) to perform optimally.
    """)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    st.write("Features scaled")

    # Step 9: Model Training and Evaluation
    st.subheader("Step 9: Model Training and Evaluation")
    st.markdown("""
    I trained the three models: Gaussian Naive Bayes, Logistic Regression, and Decision Tree. Each model is evaluated on its performance to determine the best one.
    """)

    # Gaussian Naive Bayes
    st.write("#### Gaussian Naive Bayes")
    gnb = GaussianNB()
    gnb.fit(X_train_scaled, y_train_resampled)
    y_pred_gnb = gnb.predict(X_test_scaled)
    y_prob_gnb = gnb.predict_proba(X_test_scaled)[:, 1]
    st.text(classification_report(y_test, y_pred_gnb))
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_prob_gnb))

    # Logistic Regression
    st.write("#### Logistic Regression")
    lr = LogisticRegression(class_weight='balanced', random_state=42)
    lr.fit(X_train_scaled, y_train_resampled)
    y_pred_lr = lr.predict(X_test_scaled)
    y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
    st.text(classification_report(y_test, y_pred_lr))
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_prob_lr))

    # Decision Tree
    st.write("#### Decision Tree")
    dt = DecisionTreeClassifier(class_weight='balanced', random_state=42)
    dt.fit(X_train_scaled, y_train_resampled)
    y_pred_dt = dt.predict(X_test_scaled)
    y_prob_dt = dt.predict_proba(X_test_scaled)[:, 1]
    st.text(classification_report(y_test, y_pred_dt))
    st.write("ROC AUC Score:", roc_auc_score(y_test, y_prob_dt))

    # Step 10: Model Saving
    st.subheader("Step 10: Model Saving")
    st.markdown("""
    The best-performing model is saved for future use. This enables deploying the model to make predictions on new data.
    """)
    best_model = lr
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    st.write("Best model and scaler saved as best_model.pkl and scaler.pkl")

elif page == "Model Prediction":
    # Section: Model Prediction
    st.header("Model Prediction")
    st.markdown("""
    This section allows you to input new data and get predictions using the trained model.
    """)

    # Load the trained model and scaler
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')

    st.write("#### Input New Data for Prediction")
    REVIEWS = st.selectbox("REVIEWS", [0, 1])
    BRO_TOGETHER = st.selectbox("BRO_TOGETHER", [0, 1])
    COMPARE_SIMILAR = st.selectbox("COMPARE_SIMILAR", [0, 1])
    WARRANTY = st.selectbox("WARRANTY", [0, 1])
    SPONSORED_LINKS = st.selectbox("SPONSORED_LINKS", [0, 1])

    # Make a prediction
    if st.button("Predict"):
        input_data = pd.DataFrame({
            'REVIEWS': [REVIEWS],
            'BRO_TOGETHER': [BRO_TOGETHER],
            'COMPARE_SIMILAR': [COMPARE_SIMILAR],
            'WARRANTY': [WARRANTY],
            'SPONSORED_LINKS': [SPONSORED_LINKS]
        })
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)[0]
        prediction_proba = model.predict_proba(input_data_scaled)[0][1]
        
        st.write("#### Prediction")
        st.write("Will the customer buy?", "Yes" if prediction == 1 else "No")
        st.write("Probability of purchase:", prediction_proba)
