import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

# === CONFIGURATION ===
st.set_page_config(layout="wide")
st.title("📎 ChurnSight Dashboard")

# Load trained ANN model
model = load_model('/content/ann_churn_model.h5')  # Replace with your actual model path

# Category Mappings
gender_map = {'Male': 1, 'Female': 0}
subscription_map = {'Basic': 0, 'Standard': 1, 'Premium': 2}
contract_map = {'Monthly': 0, 'Quarterly': 1, 'Annual': 2}

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("Upload test dataset (CSV, without 'Churn' column):", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    original_df = df.copy()

    # Encode categorical variables
    df['Gender'] = df['Gender'].map(gender_map)
    df['Subscription Type'] = df['Subscription Type'].map(subscription_map)
    df['Contract Length'] = df['Contract Length'].map(contract_map)

    # Feature columns
    features = ['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls',
                'Payment Delay', 'Subscription Type', 'Contract Length',
                'Total Spend', 'Last Interaction']

    # Normalize input data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])

    # Predict churn
    predictions = model.predict(X_scaled)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    # Add results
    original_df['Churn Probability'] = predictions.flatten()
    original_df['Predicted Churn'] = predicted_classes

    # === DISPLAY RESULTS ===
    st.subheader("📋Result Outputs")
    st.dataframe(original_df.head(20))

    csv = original_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", data=csv, file_name="churn_predictions.csv")

    # === DASHBOARD VISUALS ===
    st.subheader("📊 Churn Overview")
    churned = original_df[original_df['Predicted Churn'] == 1]

    col1, col2 = st.columns(2)

    with col1:
        # Pie chart by gender
        fig1 = px.pie(churned, names='Gender', title='Churned Customers by Gender')
        st.plotly_chart(fig1, use_container_width=True)

        # Bar chart by subscription type
        sub_counts = churned['Subscription Type'].value_counts().reset_index()
        sub_counts.columns = ['Subscription Type', 'Count']
        fig2 = px.bar(sub_counts, x='Subscription Type', y='Count', title='Churn by Subscription Type')
        st.plotly_chart(fig2, use_container_width=True)

        # Boxplot - Support Calls
        fig3 = px.box(original_df, x='Predicted Churn', y='Support Calls',
                      title='Support Calls vs Predicted Churn')
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Histogram of age
        fig4 = px.histogram(churned, x='Age', nbins=20, title='Age Distribution of Churned Customers')
        st.plotly_chart(fig4, use_container_width=True)

        # Bar chart by contract length
        contract_counts = churned['Contract Length'].value_counts().reset_index()
        contract_counts.columns = ['Contract Length', 'Count']
        fig5 = px.bar(contract_counts, x='Contract Length', y='Count', title='Churn by Contract Length')
        st.plotly_chart(fig5, use_container_width=True)

        # Boxplot - Total Spend
        fig6 = px.box(original_df, x='Predicted Churn', y='Total Spend',
                      title='Total Spend vs Predicted Churn')
        st.plotly_chart(fig6, use_container_width=True)

    # Violin plot - Payment Delay
    st.subheader("📈 Additional Insights")
    fig7 = px.violin(original_df, x='Predicted Churn', y='Payment Delay', box=True,
                     title='Payment Delay vs Predicted Churn')
    st.plotly_chart(fig7, use_container_width=True)

    # === ADVANCED ANALYTICS ===
    st.subheader("🧠 Advanced Churn Analytics")

    st.markdown("### 🔍 Correlation with Predicted Churn")
    corr = original_df.select_dtypes(include='number').corr()['Predicted Churn'].sort_values(ascending=False)
    st.dataframe(corr.to_frame(name="Correlation with Churn"))

    st.markdown("### 📌 High-Risk Customer Segments")
    seg_df = original_df.groupby(['Subscription Type', 'Contract Length'])['Predicted Churn'].mean().reset_index()
    seg_df.columns = ['Subscription Type', 'Contract Length', 'Churn Rate']
    st.dataframe(seg_df)

    st.markdown("### 💰 Spend by Customer Segments")
    fig8 = px.box(original_df, x='Subscription Type', y='Total Spend',
                  color='Predicted Churn', title='Spend by Subscription Type and Churn')
    st.plotly_chart(fig8, use_container_width=True)

    st.success("✅ Dashboard generated successfully.")

else:
    st.info("📎 Please upload a CSV file to begin.")
