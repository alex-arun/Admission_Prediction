import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="UCLA Admission Predictor",
    page_icon="🎓",
    layout="wide"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #2E86C1;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 0.15rem 1.75rem 0 rgba(58, 59, 69, 0.15);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.title("🎓 UCLA Graduate Admission Predictor")
st.markdown("""
Enter your academic details below to predict your chances of admission to UCLA graduate programs.
All fields are required for accurate prediction.
""")
st.markdown("---")

# Function to load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('admission_predict.csv')
    df = df.rename(columns={
        'GRE Score': 'GRE',
        'TOEFL Score': 'TOEFL',
        'LOR ': 'LOR',
        'Chance of Admit ': 'Probability'
    })
    return df

# Load the data
df = load_data()

# Create two columns for input fields
col1, col2 = st.columns(2)

# Input form
with st.form("prediction_form"):
    st.subheader("📝 Enter Your Details")
    
    with col1:
        gre = st.number_input(
            "GRE Score",
            min_value=260,
            max_value=340,
            value=300,
            help="Enter your GRE score (260-340)"
        )
        
        toefl = st.number_input(
            "TOEFL Score",
            min_value=0,
            max_value=120,
            value=100,
            help="Enter your TOEFL score (0-120)"
        )
        
        univ_rating = st.selectbox(
            "University Rating",
            options=[1, 2, 3, 4, 5],
            help="Rate your current university (1-5)"
        )
        
        cgpa = st.number_input(
            "CGPA",
            min_value=6.0,
            max_value=10.0,
            value=8.0,
            step=0.1,
            help="Enter your CGPA (6.0-10.0)"
        )
        
    with col2:
        sop = st.slider(
            "Statement of Purpose Rating",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Rate your SOP (1-5)"
        )
        
        lor = st.slider(
            "Letter of Recommendation Strength",
            min_value=1.0,
            max_value=5.0,
            value=3.0,
            step=0.5,
            help="Rate your LOR strength (1-5)"
        )
        
        research = st.radio(
            "Research Experience",
            options=["Yes", "No"],
            help="Do you have research experience?"
        )
    
    submitted = st.form_submit_button("Predict My Chances")

# Prediction section
if submitted:
    try:
        # Prepare the model
        X = df.drop(['Serial No.', 'Probability'], axis=1)
        y = df['Probability']
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train the model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Prepare input data
        research_int = 1 if research == "Yes" else 0
        input_data = np.array([[gre, toefl, univ_rating, sop, lor, cgpa, research_int]])
        
        # Scale the input data using the same scaler
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        
        # Display results
        st.markdown("### 🎯 Prediction Results")
        
        # Create three columns for displaying results
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Gauge chart for prediction
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prediction * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Admission Probability", 'font': {'size': 24}},
                delta={'reference': 75},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2E86C1"},
                    'steps': [
                        {'range': [0, 30], 'color': "#FF9999"},
                        {'range': [30, 70], 'color': "#FFD700"},
                        {'range': [70, 100], 'color': "#90EE90"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Interpretation box
        st.markdown("""
        <div class="prediction-box">
            <h4>📊 Interpretation</h4>
        """, unsafe_allow_html=True)
        
        if prediction >= 0.8:
            st.success(f"🌟 Excellent chances! Your profile shows a {prediction*100:.1f}% probability of admission.")
        elif prediction >= 0.6:
            st.success(f"✨ Good chances! Your profile shows a {prediction*100:.1f}% probability of admission.")
        elif prediction >= 0.4:
            st.warning(f"⚠️ Moderate chances. Your profile shows a {prediction*100:.1f}% probability of admission.")
        else:
            st.error(f"⚠️ Lower chances. Your profile shows a {prediction*100:.1f}% probability of admission.")
        
        # Recommendations based on the prediction
        st.markdown("### 💡 Recommendations")
        recommendations = []
        
        if gre < 320:
            recommendations.append("Consider retaking GRE to improve your score above 320")
        if toefl < 100:
            recommendations.append("A TOEFL score above 100 could strengthen your application")
        if cgpa < 8.5:
            recommendations.append("Strong research experience and LORs could help compensate for CGPA")
        if research == "No":
            recommendations.append("Try to gain some research experience before applying")
            
        if recommendations:
            for rec in recommendations:
                st.info(rec)
        else:
            st.success("Your profile looks strong across all areas! 🎉")
            
    except Exception as e:
        st.error("An error occurred during prediction. Please check your input values and try again.")
        st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666666;'>
        Note: This prediction is based on historical data and should be used as a reference only.
        Actual admission decisions may depend on additional factors.
    </div>
""", unsafe_allow_html=True)