import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns


col1, col2, col3 = st.columns([0.2, 1, 0.2])

# # Center the image in the middle column with a specified width
with col2:
     st.image("logo.jpeg", use_container_width=True)


# st.title("Workforce Insights")

# Sidebar for app selection
# st.sidebar.title("Select One")
# app_selection = st.sidebar.selectbox("Select App", ["Single Prediction", "Prediction Using Test File"])
app_selection = st.selectbox("Select App", ["Single Prediction", "Prediction Using Test File"])

# Load the pre-trained model
with open('pipeline.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# st.image("logo.png", use_container_width=True)  
# Create three columns for layout
# col1, col2, col3 = st.columns([0.2, 5, 0.2])

# Center the image in the middle column with a size of 1x1


if app_selection == "Single Prediction":
    # Function to show prediction result
    def show_prediction(satisfaction_level, last_evaluation, number_project, average_monthly_hours,
                       time_spent_company, work_accident, promotion_last_5years, department, salary):
        sample = pd.DataFrame({
            'satisfaction_level': [satisfaction_level],
            'last_evaluation': [last_evaluation],
            'number_project': [number_project],
            'average_montly_hours': [average_monthly_hours],
            'time_spend_company': [time_spent_company],
            'Work_accident': [work_accident],
            'promotion_last_5years': [promotion_last_5years],
            'departments': [department],
            'salary': [salary]
        })

        result = pipeline.predict(sample)

        if result[0] == 1:
            st.write("An employee may leave the organization.")
        else:
            st.write("An employee may stay with the organization.")

    # Streamlit app for single prediction
    

    # Employee data input fields
    satisfaction_level = st.number_input("Employee Satisfaction Level", min_value=0.0, max_value=1.0, step=0.01)
    last_evaluation = st.number_input("Last Evaluation Score", min_value=0.0, max_value=1.0, step=0.01)
    number_project = st.number_input("Number of Projects Assigned", min_value=1, step=1)
    average_monthly_hours = st.number_input("Average Monthly Hours Worked", min_value=1, step=1)
    time_spent_company = st.number_input("Time Spent at the Company (Years)", min_value=1, step=1)
    work_accident = st.radio("Work Accident (1=Yes, 0=No)", [0, 1])
    promotion_last_5years = st.radio("Promotion in Last 5 Years (1=Yes, 0=No)", [0, 1])
    department = st.selectbox("Department Name", ['sales', 'technical', 'support', 'IT', 'product_mng', 'marketing', 'RandD', 'accounting', 'hr', 'management'])
    salary = st.selectbox("Salary Category", ['low', 'medium', 'high'])

    # Predict button
    if st.button("Predict"):
        show_prediction(satisfaction_level, last_evaluation, number_project, average_monthly_hours,
                        time_spent_company, work_accident, promotion_last_5years, department, salary)
else:
    def process_data(data):
        result = pipeline.predict(data)
        data['Predicted_target'] = ["An employee may leave." if pred == 1 else "An employee may stay." for pred in result]
        return data

    def create_visualizations(data):
        st.subheader("Data Analysis and Visualizations")
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        sns.histplot(data['satisfaction_level'], bins=20, kde=True, ax=axs[0, 0])
        axs[0, 0].set_title('Distribution of Employee Satisfaction Level')

        sns.histplot(data['last_evaluation'], bins=20, kde=True, ax=axs[0, 1])
        axs[0, 1].set_title('Distribution of Last Evaluation Score')

        if 'Predicted_target' in data.columns:
            sns.countplot(data=data, x='Predicted_target', palette='Set2', ax=axs[1, 0])
            axs[1, 0].set_title('Predicted Employee Churn')

            sns.boxplot(data=data, x='Predicted_target', y='satisfaction_level', palette='Set2', ax=axs[1, 1])
            axs[1, 1].set_title('Satisfaction Level by Prediction')

        plt.tight_layout()
        st.pyplot(fig)

        # Pie charts with error handling
        fig2, axs2 = plt.subplots(1, 2, figsize=(14, 6))

        if not data['departments'].empty:
            department_counts = data['departments'].value_counts()
            axs2[0].pie(department_counts, labels=department_counts.index, autopct='%1.1f%%', startangle=140)
            axs2[0].set_title('Department Distribution')

        if not data['salary'].empty:
            salary_counts = data['salary'].value_counts()
            axs2[1].pie(salary_counts, labels=salary_counts.index, autopct='%1.1f%%', startangle=140)
            axs2[1].set_title('Salary Distribution')

        plt.tight_layout()
        st.pyplot(fig2)

        # Handling missing categories in pie charts
        fig3, axs3 = plt.subplots(1, 2, figsize=(14, 6))
        work_accident_counts = data['Work_accident'].value_counts()
        work_labels = ['No Accident', 'Accident'][:len(work_accident_counts)]
        axs3[0].pie(work_accident_counts, labels=work_labels, autopct='%1.1f%%', startangle=140)
        axs3[0].set_title('Work Accident Distribution')

        promotion_counts = data['promotion_last_5years'].value_counts()
        promotion_labels = ['No Promotion', 'Promotion'][:len(promotion_counts)]
        axs3[1].pie(promotion_counts, labels=promotion_labels, autopct='%1.1f%%', startangle=140)
        axs3[1].set_title('Promotion in Last 5 Years Distribution')

        plt.tight_layout()
        st.pyplot(fig3)

        # Count plot for number of projects assigned
        plt.figure(figsize=(10, 6))
        sns.countplot(data=data, x='number_project', palette='Set2')
        plt.title('Number of Projects Assigned')
        plt.xlabel('Number of Projects')
        plt.ylabel('Count')
        st.pyplot(plt)

        # Box plot for average monthly hours worked by predicted target
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=data, x='Predicted_target', y='average_montly_hours', palette='Set2')
        plt.title('Average Monthly Hours by Prediction')
        plt.xlabel('Prediction')
        plt.ylabel('Average Monthly Hours')
        st.pyplot(plt)

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            data.columns = data.columns.str.replace('\n', '')
            data.rename(columns={'Departments ': 'departments'}, inplace=True)
            data = data.drop_duplicates()
            processed_data = process_data(data)
            st.write("Processed Data:")
            st.write(processed_data)
            create_visualizations(processed_data)
            processed_data.to_csv('processed_data.csv', index=False)
            st.success("Data saved successfully as 'processed_data.csv'!")
        except Exception as e:
            st.error(f"Failed to process the file: {e}")
