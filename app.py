import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.title("Student Grade Prediction System")

# Function to preprocess data
def preprocess_data(data):
    # Handling missing values for numeric columns only
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
    
    # Encoding categorical variables
    categorical_columns = data.select_dtypes(include=[object]).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    
    return data

# Function to classify the predicted grades
def classify_grade(grade):
    if grade >= 75:
        return "Distinction"
    elif grade >= 60:
        return "First Class"
    elif grade >= 50:
        return "Second Class"
    elif grade >= 40:
        return "Pass"
    else:
        return "Fail"

# Initial Upload Data Section
st.subheader("Step 1: Upload Dataset")
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# Upload dataset and generate results
if uploaded_file:
    # Read and display the uploaded dataset preview
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())
    
    # Button to start generating results
    if st.button("Generate Predictions"):
        # Preprocess the data
        data = preprocess_data(df)
        
        # Set target variable (e.g., "math score", "reading score", or "writing score")
        target_column = "math score"  # Change this line if you want to predict other scores
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train a model
        model = LinearRegression()  # You can change this to DecisionTreeRegressor or RandomForestRegressor
        model.fit(X_train, y_train)
        
        # Predict the grades for the full dataset
        predictions = model.predict(X)
        data["Predicted Grade"] = predictions  # Add predictions to the dataframe
        data["Grade Classification"] = data["Predicted Grade"].apply(classify_grade)  # Classify the predicted grades
        
        # Model performance metrics
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Display performance metrics
        st.subheader("Step 2: Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R-Squared (R²): {r2:.2f}")
        
        # Show the results
        st.subheader("Step 3: Predicted Grades and Classifications")
        st.dataframe(data[["Predicted Grade", "Grade Classification"]])
        
        # Button to download the results
        def convert_df(df):
            """Converts the DataFrame to CSV format for download."""
            return df.to_csv(index=False).encode('utf-8')

        csv = convert_df(data[["Predicted Grade", "Grade Classification"]])

        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="predicted_grades.csv",
            mime="text/csv"
        )

        # Button to go back to upload data
        if st.button("Go Back to Upload Data"):
            st.session_state["data_uploaded"] = False
            st.experimental_rerun()

else:
    st.warning("Please upload a CSV file to proceed.")
