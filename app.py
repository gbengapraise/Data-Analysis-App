import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import io
st.title("📊 Welcome to the Math Graph & Data Analysis App!")

st.markdown(
    """
    <div style='text-align: center; font-size: 20px; color: #4CAF50;'>
    Unlock the power of your data! Upload, visualize, analyze, and predict with ease.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")

st.header("✨ Features")
st.markdown(
    """
    - **Upload Datasets** (CSV, Excel, JSON)
    - **Multiple Uploads** allowed
    - **Data Cleaning Tools** (handle missing values, duplicates)
    - **Interactive Data Visualizations** (Bar, Line, Pie, Heatmaps)
    - **Machine Learning Predictions** (Linear Regression)
    - **Beautiful, Smooth UI** (Light and Dark Mode support)
    """
)

st.write("---")

st.subheader("💡 Why Data Matters?")
st.markdown(
    """
    > *"Without data, you're just another person with an opinion."*  
    > — W. Edwards Deming
    """
)

st.write("---")

st.success("✅ Start by selecting an option from the sidebar!")
# Function to handle file upload and return DataFrame
def upload_files():
    uploaded_files = st.file_uploader("Upload your CSV or Excel files", type=["csv", "xlsx"], accept_multiple_files=True)
    dataframes = []

    if uploaded_files:
        for file in uploaded_files:
            if file.type == "text/csv":
                df = pd.read_csv(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(file)
            dataframes.append(df)
    return dataframes

# Function to display statistics and visualizations
def show_statistics(df):
    st.subheader("Basic Statistics")
    st.write(df.describe())

    st.subheader("Data Preview")
    st.write(df.head())

    # Visualizations
    st.subheader("Data Visualizations")
    visualization_option = st.selectbox("Choose Visualization", ["Correlation Heatmap", "Pairplot", "Boxplot"])

    if visualization_option == "Correlation Heatmap":
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    elif visualization_option == "Pairplot":
        fig = sns.pairplot(df)
        st.pyplot(fig)

    elif visualization_option == "Boxplot":
        numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
        selected_column = st.selectbox("Select column for boxplot", numerical_columns)
        fig = plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[selected_column])
        st.pyplot(fig)

# Function for predictive analysis
def predictive_analysis(df):
    st.subheader("Predictive Analysis")

    # Ensure there are at least two numerical columns for linear regression
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_columns) < 2:
        st.warning("Not enough numerical columns for predictive analysis.")
        return

    # Select dependent and independent variables
    dependent_var = st.selectbox("Select Dependent Variable", numerical_columns)
    independent_vars = st.multiselect("Select Independent Variables", numerical_columns.drop(dependent_var))

    if dependent_var and independent_vars:
        X = df[independent_vars]
        y = df[dependent_var]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse}")

        # Plot predictions vs true values
        st.subheader("Predictions vs True Values")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_test, y_pred)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        st.pyplot(fig)

# Main app interface
def main():
    st.title("Data Analysis Dashboard")

    # File upload section
    st.subheader("Upload Data")
    dataframes = upload_files()

    if dataframes:
        for df in dataframes:
            st.write(f"Data from file: {df.shape[0]} rows and {df.shape[1]} columns")
            show_statistics(df)

            # Button for Predictive Analysis
            if st.button("Run Predictive Analysis"):
                predictive_analysis(df)

            # Button for visualizations
            if st.button("Generate Visualizations"):
                show_statistics(df)

    else:
        st.warning("Please upload a dataset to proceed.")

if __name__ == "__main__":
    main()
