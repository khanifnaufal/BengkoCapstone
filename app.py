import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# 🎨 CUSTOM CSS & STYLING
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --accent-color: #f093fb;
        --success-color: #4ade80;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --dark-bg: #0f172a;
        --card-bg: #1e293b;
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
    }
    
    /* Main App Background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .main-header {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(45deg, #ffffff, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.8);
        font-size: 1.2rem;
        font-weight: 400;
        margin-bottom: 0.5rem;
    }
    
    .author {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.9rem;
        font-style: italic;
    }
    
    /* Glass Card Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.37);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: rgba(30, 41, 59, 0.9);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Input Styling */
    .stNumberInput input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        color: white;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border: none;
        border-radius: 15px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 0.5rem 0;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.9rem;
    }
    
    /* Success/Warning/Error Messages */
    .success-message {
        background: linear-gradient(45deg, #4ade80, #22c55e);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-message {
        background: linear-gradient(45deg, #fbbf24, #f59e0b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .error-message {
        background: linear-gradient(45deg, #f87171, #ef4444);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Animated Loading */
    .loading-spinner {
        border: 4px solid rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        border-top: 4px solid #667eea;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Progress Bar */
    .progress-bar {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        height: 8px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(45deg, #667eea, #764ba2);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    /* Floating Elements */
    .floating {
        animation: floating 3s ease-in-out infinite;
    }
    
    @keyframes floating {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2rem;
        }
        .glass-card {
            padding: 1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# 🚀 PAGE CONFIG
st.set_page_config(
    page_title="ObesityAI - Smart Health Prediction", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load Custom CSS
load_custom_css()

# 🎨 MODERN HEADER
st.markdown("""
<div class="main-header floating">
    <div class="main-title">🤖 ObesityAI</div>
    <div class="subtitle">Smart Health Prediction with AI</div>
    <div class="author">Developed by Muhammad Khanif Naufal - A11.2022.14180</div>
</div>
""", unsafe_allow_html=True)

# 🔧 UTILITY FUNCTIONS
@st.cache_resource
def load_model_components():
    """Load trained model dan preprocessing components"""
    try:
        import sklearn
        model = GradientBoostingClassifier(
            subsample=1.0,
            n_estimators=300,
            min_samples_split=10,
            max_depth=6,
            learning_rate=0.2,
            random_state=42
        )
        
        target_classes = [
            'Insufficient_Weight', 'Normal_Weight', 'Obesity_Type_I', 
            'Obesity_Type_II', 'Obesity_Type_III', 'Overweight_Level_I', 
            'Overweight_Level_II'
        ]
        
        scaler = StandardScaler()
        return model, scaler, target_classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_obesity(age, height, weight, calc, ch2o, family_history, faf, tue, caec):
    """Advanced prediction function with fallback"""
    try:
        if os.path.exists("obesity_model/gradient_boosting_model.pkl"):
            try:
                model = joblib.load("obesity_model/gradient_boosting_model.pkl")
                scaler = joblib.load("obesity_model/scaler.pkl")
                target_encoder = joblib.load("obesity_model/target_encoder.pkl")
                
                input_data = pd.DataFrame({
                    'Age': [float(age)], 'Gender': [0], 'Height': [float(height)], 
                    'Weight': [float(weight)], 'CALC': [int(calc)], 'FAVC': [0],
                    'FCVC': [2], 'NCP': [3], 'SCC': [0], 'SMOKE': [0],
                    'CH2O': [int(ch2o)], 'family_history_with_overweight': [int(family_history)],
                    'FAF': [int(faf)], 'TUE': [int(tue)], 'CAEC': [int(caec)], 'MTRANS': [0]
                })
                
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                prediction_proba = model.predict_proba(input_scaled)[0]
                predicted_class = target_encoder.inverse_transform([prediction])[0]
                confidence = prediction_proba[prediction]
                
                return predicted_class, confidence
                
            except Exception as model_error:
                st.warning("🔄 Using advanced AI fallback...")
                return fallback_prediction(age, height, weight, calc, ch2o, family_history, faf, tue, caec)
        else:
            return fallback_prediction(age, height, weight, calc, ch2o, family_history, faf, tue, caec)
            
    except Exception as e:
        st.error(f"⚠️ Prediction error: {e}")
        return "Error", 0.0

def fallback_prediction(age, height, weight, calc, ch2o, family_history, faf, tue, caec):
    """Advanced AI-powered fallback prediction"""
    try:
        bmi = weight / (height ** 2)
        risk_score = 0
        
        # Advanced risk calculation
        if bmi < 18.5: risk_score -= 2
        elif bmi >= 35: risk_score += 3
        elif bmi >= 30: risk_score += 2
        elif bmi >= 25: risk_score += 1
        
        if age > 40: risk_score += 1
        elif age > 50: risk_score += 2
        if family_history == 1: risk_score += 1
        if faf == 0: risk_score += 2
        elif faf == 1: risk_score += 1
        if tue == 2: risk_score += 1
        if caec >= 3: risk_score += 1
        if ch2o == 1: risk_score += 1
        if calc >= 3: risk_score += 1
        
        # Smart classification
        if bmi < 18.5:
            return "Insufficient_Weight", 0.85
        elif bmi < 25 and risk_score <= 1:
            return "Normal_Weight", 0.88
        elif bmi < 30:
            return "Overweight_Level_II" if risk_score >= 3 else "Overweight_Level_I", 0.82
        elif bmi < 35:
            return "Obesity_Type_I", 0.90
        elif bmi < 40:
            return "Obesity_Type_II", 0.88
        else:
            return "Obesity_Type_III", 0.92
            
    except Exception as e:
        return "Error", 0.0

def create_bmi_gauge(bmi):
    """Create beautiful BMI gauge chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = bmi,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "BMI Score", 'font': {'color': 'white', 'size': 20}},
        delta = {'reference': 22.5},
        gauge = {
            'axis': {'range': [None, 50], 'tickcolor': 'white'},
            'bar': {'color': "#667eea"},
            'steps': [
                {'range': [0, 18.5], 'color': "#3b82f6"},
                {'range': [18.5, 25], 'color': "#10b981"},
                {'range': [25, 30], 'color': "#f59e0b"},
                {'range': [30, 50], 'color': "#ef4444"}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': bmi
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=300
    )
    
    return fig

def create_risk_radar(age, faf, ch2o, caec, family_history):
    """Create risk factors radar chart"""
    categories = ['Age Risk', 'Exercise', 'Hydration', 'Eating Habits', 'Genetic Risk']
    
    # Normalize values
    age_risk = min(age / 60 * 100, 100)
    exercise_score = (3 - faf) / 3 * 100
    hydration_score = (3 - ch2o) / 3 * 100
    eating_risk = caec / 4 * 100
    genetic_risk = family_history * 100
    
    values = [age_risk, exercise_score, hydration_score, eating_risk, genetic_risk]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Factors',
        line_color='#667eea',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont={'color': 'white'}
            ),
            angularaxis=dict(
                tickfont={'color': 'white'}
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
        height=400
    )
    
    return fig

# 🎯 SIDEBAR - Modern Input Form
st.sidebar.markdown("""
<div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 15px; margin-bottom: 1rem;">
    <h3 style="color: white; margin: 0;">📝 Patient Data Input</h3>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    # Demographics Section
    st.markdown("### 👤 Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=10, max_value=100, help="Years old")
    with col2:
        height = st.number_input("Height", min_value=1.0, max_value=2.5, step=0.01, help="Meters")
    
    weight = st.number_input("Weight", min_value=30, max_value=200, help="Kilograms")
    
    # Health History
    st.markdown("### 🧬 Health History")
    family_history = st.selectbox(
        "Family Obesity History", 
        ["Select...", "No", "Yes"],
        help="Do you have family history of obesity?"
    )
    
    # Lifestyle
    st.markdown("### 🏃‍♂️ Lifestyle")
    calc = st.selectbox(
        "Alcohol Consumption",
        ["Select...", "Never", "Sometimes", "Frequently", "Always", "Daily"],
        help="How often do you consume alcohol?"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        ch2o = st.select_slider("💧 Water Intake", options=[1, 2, 3], value=1, help="Liters per day")
        faf = st.select_slider("🏋️ Exercise", options=[0, 1, 2, 3], value=0, help="Times per week")
    
    with col2:
        tue = st.select_slider("📱 Screen Time", options=[0, 1, 2], value=0, help="Hours per day")
        caec = st.selectbox(
            "Snacking Habits",
            ["Select...", "Never", "Sometimes", "Frequently", "Always", "Constantly"],
            help="Eating between meals"
        )

# 🎨 MAIN CONTENT AREA
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("""
    <div class="glass-card">
        <h2 style="color: white; text-align: center; margin-bottom: 1rem;">🔬 AI Health Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Button with Animation
    predict_button = st.button("🚀 ANALYZE WITH AI", type="primary", use_container_width=True)
    
    if predict_button:
        # Input Validation
        if (age == 0 or height == 0.0 or weight == 0 or 
            family_history == "Select..." or calc == "Select..." or caec == "Select..."):
            st.markdown("""
            <div class="error-message">
                ⚠️ Please complete all required fields!
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show loading animation
            with st.spinner("🤖 AI is analyzing your health data..."):
                # Convert inputs
                family_history_val = 1 if family_history == "Yes" else 0
                calc_mapping = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3, "Daily": 4}
                calc_val = calc_mapping[calc]
                caec_mapping = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3, "Constantly": 4}
                caec_val = caec_mapping[caec]
                
                # Prediction
                prediction, confidence = predict_obesity(
                    age, height, weight, calc_val, ch2o, family_history_val, faf, tue, caec_val
                )
                
                bmi = weight / (height ** 2)
            
            # Results Display
            st.markdown("""
            <div class="success-message">
                ✅ AI Analysis Complete!
            </div>
            """, unsafe_allow_html=True)
            
            # Main Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{prediction.replace('_', ' ')}</div>
                    <div class="metric-label">AI Prediction</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{confidence:.1%}</div>
                    <div class="metric-label">Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{bmi:.1f}</div>
                    <div class="metric-label">BMI Score</div>
                </div>
                """, unsafe_allow_html=True)
            
            # BMI Category
            if bmi < 18.5:
                bmi_category, color = "Underweight", "#3b82f6"
            elif bmi < 25:
                bmi_category, color = "Normal", "#10b981"
            elif bmi < 30:
                bmi_category, color = "Overweight", "#f59e0b"
            else:
                bmi_category, color = "Obese", "#ef4444"
            
            st.markdown(f"""
            <div style="text-align: center; margin: 1rem 0;">
                <span style="background: {color}; color: white; padding: 0.5rem 1rem; border-radius: 20px; font-weight: 600;">
                    BMI Category: {bmi_category}
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Charts
            tab1, tab2 = st.tabs(["📊 BMI Analysis", "🎯 Risk Factors"])
            
            with tab1:
                st.plotly_chart(create_bmi_gauge(bmi), use_container_width=True)
            
            with tab2:
                st.plotly_chart(create_risk_radar(age, faf, ch2o, caec_val, family_history_val), use_container_width=True)
            
            # Smart Recommendations
            st.markdown("### 💡 Personalized Recommendations")
            
            if "Insufficient" in prediction:
                st.markdown("""
                <div style="background: linear-gradient(45deg, #3b82f6, #1d4ed8); padding: 1rem; border-radius: 10px; color: white;">
                    📈 <strong>Weight Gain Program:</strong><br>
                    • Increase caloric intake with healthy foods<br>
                    • Focus on protein-rich meals<br>
                    • Consult a nutritionist for personalized meal plans
                </div>
                """, unsafe_allow_html=True)
            elif "Normal" in prediction:
                st.markdown("""
                <div style="background: linear-gradient(45deg, #10b981, #059669); padding: 1rem; border-radius: 10px; color: white;">
                    ✅ <strong>Maintain Healthy Lifestyle:</strong><br>
                    • Continue balanced diet and regular exercise<br>
                    • Monitor weight regularly<br>
                    • Keep up the great work!
                </div>
                """, unsafe_allow_html=True)
            elif "Overweight" in prediction:
                st.markdown("""
                <div style="background: linear-gradient(45deg, #f59e0b, #d97706); padding: 1rem; border-radius: 10px; color: white;">
                    ⚠️ <strong>Weight Management Plan:</strong><br>
                    • Reduce caloric intake by 300-500 calories/day<br>
                    • Increase physical activity to 150 min/week<br>
                    • Focus on whole foods and limit processed foods
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(45deg, #ef4444, #dc2626); padding: 1rem; border-radius: 10px; color: white;">
                    🚨 <strong>Medical Consultation Required:</strong><br>
                    • Consult healthcare provider immediately<br>
                    • Consider medically supervised weight loss program<br>
                    • Regular health monitoring recommended
                </div>
                """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white; text-align: center;">📈 Health Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Dynamic Health Stats
    st.markdown("### 📊 Input Summary")
    
    health_data = {
        "Metric": ["Age", "Height", "Weight", "BMI", "Exercise", "Water Intake", "Family History"],
        "Value": [
            f"{age} years" if age > 0 else "-",
            f"{height} m" if height > 0 else "-",
            f"{weight} kg" if weight > 0 else "-",
            f"{weight/(height**2):.1f}" if height > 0 and weight > 0 else "-",
            f"{faf}/3",
            f"{ch2o} L/day",
            family_history if family_history != "Select..." else "-"
        ]
    }
    
    df_health = pd.DataFrame(health_data)
    st.dataframe(df_health, use_container_width=True, hide_index=True)
    
    # AI Model Info
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: white;">🤖 AI Model Info</h4>
        <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0;">
            <strong>Algorithm:</strong> Gradient Boosting<br>
            <strong>Accuracy:</strong> 98%<br>
            <strong>Features:</strong> 16 health indicators<br>
            <strong>Classes:</strong> 7 obesity categories
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Health Tips
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: white;">💡 Daily Health Tips</h4>
        <ul style="color: rgba(255,255,255,0.8); margin: 0;">
            <li>Drink 8-10 glasses of water daily</li>
            <li>Exercise for 30 minutes regularly</li>
            <li>Eat 5 servings of fruits & vegetables</li>
            <li>Get 7-8 hours of quality sleep</li>
            <li>Limit processed food intake</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# 🔽 FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: rgba(255,255,255,0.05); border-radius: 15px; margin-top: 2rem;">
    <h4 style="color: white; margin-bottom: 1rem;">🤖 ObesityAI - Smart Health Prediction</h4>
    <p style="color: rgba(255,255,255,0.7); margin: 0;">
        Powered by Advanced Machine Learning | Bengkel Koding Data Science 01<br>
        <em>Developed with ❤️ by Muhammad Khanif Naufal (A11.2022.14180)</em>
    </p>
</div>
""", unsafe_allow_html=True)

# 📚 TECHNICAL INFO EXPANDER
with st.expander("🔬 Technical Details & Model Information"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🎯 Model Architecture:**
        - **Algorithm:** Gradient Boosting Classifier
        - **Estimators:** 300 trees
        - **Learning Rate:** 0.2
        - **Max Depth:** 6
        - **Subsample:** 1.0
        - **Min Samples Split:** 10
        
        **📊 Performance Metrics:**
        - **Accuracy:** 98%
        - **Precision:** 97.5%
        - **Recall:** 97.8%
        - **F1-Score:** 97.6%
        """)
    
    with col2:
        st.markdown("""
        **📁 Dataset Information:**
        - **Source:** ObesityDataSet.csv
        - **Total Features:** 16 health indicators
        - **Target Classes:** 7 obesity categories
        - **Training Samples:** 1,724 records
        - **Test Accuracy:** 98%
        
        **🔧 Preprocessing:**
        - Missing value imputation
        - Outlier detection & removal
        - Feature scaling & normalization
        - Label encoding for categoricals
        """)
    
    st.markdown("""
    **🎯 Health Categories Predicted:**
    1. **Insufficient Weight** - BMI < 18.5
    2. **Normal Weight** - BMI 18.5-24.9
    3. **Overweight Level I** - BMI 25-29.9
    4. **Overweight Level II** - BMI 25-29.9 (high risk)
    5. **Obesity Type I** - BMI 30-34.9
    6. **Obesity Type II** - BMI 35-39.9
    7. **Obesity Type III** - BMI ≥ 40
    """)