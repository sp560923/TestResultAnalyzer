import streamlit as st
import google.generativeai as genai
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Initialize Streamlit app
st.set_page_config(page_title="Gemini-Pro Data Analysis Tool")
st.header("Data Analysis, Trend Forecasting, and Q&A with Gemini-Pro")

# Input for Google API key
GOOGLE_API_KEY = st.text_input("Enter your Google API Key:")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])

    # File uploader for CSV/Excel files
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])

    # Input for text-based questions to Gemini-pro
    input_text = st.text_input("Ask a question for the Gemini model:")
    submit_question = st.button("Submit Question")

    # Function to handle data analysis
    def analyze_data(df):
        try:
            if 'TestResult' in df.columns:
                # Map the TestResult statuses to numeric values
                df['TestResult_numeric'] = df['TestResult'].map({'Pass': 1, 'Fail': 0, 'Blocked': -1, 'Skipped': -2})

                # Linear Regression for Pass/Fail/Blocked/Skipped Trend Analysis
                X = np.arange(len(df)).reshape(-1, 1)  # Time index based on test case order
                y = df['TestResult_numeric']  # Status mapped as numeric values

                # Section 1: Recurring Test Failures and Patterns
                st.subheader("Recurring Test Failures and Patterns")
                recurring_failures = df[df['TestResult'] == 'Fail'].groupby('Testcase_Description').size()
                if not recurring_failures.empty:
                    st.write("Failures grouped by Testcase Description:")
                    st.dataframe(recurring_failures)
                else:
                    st.write("No recurring failures found.")

                # Section 2: Trends in Pass/Fail/Blocked/Skipped Rates and Performance (Linear Regression)
                st.subheader("Trends in Pass/Fail/Blocked/Skipped Rates and Performance")
                model_lr = LinearRegression()
                model_lr.fit(X, y)
                trend = model_lr.predict(X)
                
                st.markdown("""**Trend Analysis (Linear Regression):**  
                The linear regression model is used to analyze trends in test results (Pass, Fail, Blocked, Skipped) over time.
                """)

                plt.figure(figsize=(10, 6))
                plt.scatter(X, y, color='blue', label="Actual Test Results")
                plt.plot(X, trend, color='red', linewidth=2, label="Trend (Linear Regression)")
                plt.title('Test Results Trend Analysis (Linear Regression)', fontsize=14)
                plt.xlabel('Test Case ID (Numeric)', fontsize=12)
                plt.ylabel('Test Result (Pass=1, Fail=0, Blocked=-1, Skipped=-2)', fontsize=12)
                plt.legend()
                st.pyplot(plt)

                # Forecasting future results using ARIMA
                model_arima = sm.tsa.ARIMA(df['TestResult_numeric'], order=(1, 1, 1))
                model_fit = model_arima.fit()
                forecast = model_fit.forecast(steps=5)

                st.markdown("""**Forecast for Next 5 Test Results (ARIMA):**  
                ARIMA model forecast for future test outcomes based on historical data.
                """)

                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(df)+1), df['TestResult_numeric'], label='Actual Test Results', color='blue')
                plt.plot(range(len(df)+1, len(df)+6), forecast, label='Forecast (Next 5 Results)', color='green')
                plt.title('Forecast for Next 5 Test Results (ARIMA)', fontsize=14)
                plt.xlabel('Test Case ID (Numeric) (Including Forecasted)', fontsize=12)
                plt.ylabel('Test Result (Pass=1, Fail=0, Blocked=-1, Skipped=-2)', fontsize=12)
                plt.legend()
                st.pyplot(plt)

                # Section 3: Test Results Analysis (with Pie Chart and Bar Graph for All Statuses)
                st.subheader("Test Results Analysis (Pass/Fail/Blocked/Skipped)")
                result_counts = df['TestResult'].value_counts()
                st.write("Test Results Count:")
                st.write(result_counts)

                # Pie Chart for Test Results Distribution
                fig, ax = plt.subplots()
                ax.pie(result_counts, labels=result_counts.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig)

                # Bar Graph for Test Results Count
                st.write("Test Results Distribution Bar Graph:")
                fig, ax = plt.subplots()
                ax.bar(result_counts.index, result_counts.values, color=['green', 'red', 'orange', 'gray'])
                ax.set_ylabel('Count')
                ax.set_title('Test Results Distribution (Pass/Fail/Blocked/Skipped)')
                st.pyplot(fig)

                # Additional Analysis: Breakdown by Priority and Severity
                st.subheader("Analysis by Priority and Severity")
                priority_counts = df.groupby('Priority')['TestResult'].value_counts().unstack().fillna(0)
                severity_counts = df.groupby('Severity')['TestResult'].value_counts().unstack().fillna(0)

                st.write("Priority-wise Test Result Distribution:")
                st.dataframe(priority_counts)

                st.write("Severity-wise Test Result Distribution:")
                st.dataframe(severity_counts)

            else:
                st.error("The file does not contain the required 'TestResult' column.")

        except Exception as e:
            st.error(f"An error occurred during data analysis: {str(e)}")

    # Function to interact with Gemini-pro
    def get_gemini_response(question, context="software testing"):
        try:
            # Modify the prompt to include the correct context
            prompt = f"Analyze the provided {context} data and identify recurring test failures, trends in pass/fail/blocked/skipped rates, and provide a detailed analysis."
            response = chat.send_message(prompt, stream=True)
            return response
        except Exception as e:
            st.error(f"Error interacting with Gemini-pro: {str(e)}")
            return None

    # If a file is uploaded, process it and prepare the analysis
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Uploaded Data:")
            st.dataframe(df)

            # Call the analyze_data function to perform trend analysis and create charts
            analyze_data(df)

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

    # If a question is submitted manually, interact with the Gemini-pro model
    if submit_question and input_text:
        response = get_gemini_response(input_text, context="software testing")
        
        if response:
            st.subheader("Gemini-pro Response:")
            for chunk in response:
                st.write(chunk.text)
else:
    st.warning("Please enter your Google API Key to start.")
