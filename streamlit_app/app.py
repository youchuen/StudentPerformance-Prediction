# Streamlit Application for Student Performance Prediction
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

from src.config import BIN_COLS, SELECTED_FEATURES, DEFAULT_TARGET

ROOT = Path(__file__).resolve().parent.parent  
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.csv"
MODEL_PATH   = ROOT / "outputs" / "models"

# Set page configuration
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# Set title
st.title("🎓 Student Performance Prediction System")
st.markdown("---")

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Function", [
    "Data Exploration", 
    "Prediction System", 
    "Model Evaluation", 
    "Model Interpretation", 
    "Student Profiles"
])

# Load data
@st.cache_data
def load_data():
    """Load and preprocess the dataset"""
    try:
        data = pd.read_csv(CLEANED_PATH)
        data.columns = data.columns.str.strip()  # Clean column names
        
        # Apply the same binary transformations as in training
        binary_cols = BIN_COLS
        
        for col in binary_cols:
            if col in data.columns:
                data[col] = (data[col] >= 1).astype(int)
        
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load the best model and preprocessing components
@st.cache_resource
def load_best_model():
    """Load the best trained model and preprocessing components"""
    model_dir = MODEL_PATH
    
    try:
        # Load the best model
        best_model_path = os.path.join(model_dir, 'best_model.pkl')
        if os.path.exists(best_model_path):
            model = joblib.load(best_model_path)
            st.sidebar.success(" Best model loaded successfully")
        else:
            st.sidebar.error(" Best model not found")
            return None, {}
        
        # Load preprocessing components
        components = {}
        component_files = {
            'label_encoder': 'label_encoder.pkl',
            'feature_columns': 'feature_columns.pkl',
            'numerical_columns': 'numerical_columns.pkl',
            'categorical_columns': 'categorical_columns.pkl',
            'class_names': 'class_names.pkl'
        }
        
        for component_name, file_name in component_files.items():
            file_path = os.path.join(model_dir, file_name)
            if os.path.exists(file_path):
                components[component_name] = joblib.load(file_path)
            else:
                st.sidebar.warning(f"⚠️ {component_name} not found")
        
        return model, components
        
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")
        return None, {}

# Load performance data
@st.cache_data
def load_performance_data():
    """Load model performance results and feature importance"""
    model_dir = MODEL_PATH
    performance_data = {}
    
    try:
        # Load performance results
        results_path = os.path.join(model_dir, 'model_performance_results.csv')
        if os.path.exists(results_path):
            performance_data['results'] = pd.read_csv(results_path, index_col=0)
        
        # Load feature importance
        importance_path = os.path.join(model_dir, 'feature_importance.csv')
        if os.path.exists(importance_path):
            performance_data['feature_importance'] = pd.read_csv(importance_path)
            
        return performance_data
        
    except Exception as e:
        st.sidebar.error(f"Error loading performance data: {e}")
        return {}

# Load data and model
data = load_data()
model, components = load_best_model()
performance_data = load_performance_data()
model_loaded = model is not None

# Display content based on selected page
if page == "Data Exploration":
    st.header("📊 Data Exploration")
    
    if data is not None:
        # Display basic statistics
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", data.shape[0])
        col2.metric("Total Features", data.shape[1])
        col3.metric("Target Classes", data[DEFAULT_TARGET].nunique() if DEFAULT_TARGET in data.columns else "N/A")
        
        # Display first few rows
        st.subheader("Sample Data")
        st.dataframe(data.head())
        
        # Statistical description
        st.subheader("Statistical Summary")
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        st.dataframe(data[numeric_cols].describe())
        
        # Target variable distribution
        if DEFAULT_TARGET in data.columns:
            st.subheader("Student Status Distribution")
            fig, ax = plt.subplots(figsize=(10, 6))
            target_counts = data[DEFAULT_TARGET].value_counts()
            colors = ['#ff9999', '#66b3ff', '#99ff99']
            bars = ax.bar(target_counts.index, target_counts.values, color=colors)
            ax.set_xlabel('Student Status')
            ax.set_ylabel('Number of Students')
            ax.set_title('Distribution of Student Outcomes')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        
        # Feature exploration
        st.subheader("Feature Analysis")
        available_features = [col for col in data.columns if col != DEFAULT_TARGET]
        selected_feature = st.selectbox("Select a feature to analyze", available_features)
        
        if selected_feature and DEFAULT_TARGET in data.columns:
            st.subheader(f"Relationship: {selected_feature} vs Student Status")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Feature distribution
            if data[selected_feature].nunique() < 10:
                # Categorical feature
                counts = data[selected_feature].value_counts()
                ax1.bar(counts.index, counts.values)
                ax1.set_title(f'Distribution of {selected_feature}')
            else:
                # Numerical feature
                ax1.hist(data[selected_feature].dropna(), bins=20, alpha=0.7)
                ax1.set_title(f'Distribution of {selected_feature}')
            
            # Relationship with target
            if data[selected_feature].nunique() < 10:
                crosstab = pd.crosstab(data[selected_feature], data[DEFAULT_TARGET], normalize='index')
                crosstab.plot(kind='bar', stacked=True, ax=ax2)
                ax2.set_title(f'{selected_feature} vs Student Status (Proportional)')
            else:
                sns.boxplot(x=DEFAULT_TARGET, y=selected_feature, data=data, ax=ax2)
                ax2.set_title(f'{selected_feature} by Student Status')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Correlation analysis for numerical features
        if len(numeric_cols) > 1:
            st.subheader("Feature Correlation Matrix")
            fig, ax = plt.subplots(figsize=(12, 8))
            corr_matrix = data[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                       center=0, square=True, linewidths=.5, ax=ax)
            ax.set_title('Feature Correlation Matrix')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Unable to load data. Please ensure 'cleaned.csv' exists in the current directory.")

elif page == "Prediction System":
    st.header("🎯 Student Performance Prediction")
    
    # Check if model is loaded
    if not model_loaded:
        st.warning(" Model not loaded. Please ensure the training script has been run successfully.")
        st.code("python src/modeling.py --input data/cleaned.csv --output outputs/models")
        
        # Show example prediction
        if st.button("Show Example Prediction"):
            st.info("This is a demonstration with simulated results.")
            result = np.random.choice(['Dropout', 'Graduate', 'Enrolled'], p=[0.3, 0.5, 0.2])
            
            if result == "Dropout":
                st.error("🚨 Prediction: Likely to Drop Out")
            elif result == "Graduate":
                st.success("🎓 Prediction: Likely to Graduate")
            else:
                st.info("📚 Prediction: Likely to Stay Enrolled")
    else:
        # Display model information with best model details
        if 'results' in performance_data and not performance_data['results'].empty:
            results_df = performance_data['results']
            best_model = results_df['roc_auc'].idxmax()
            best_score = results_df['roc_auc'].max()
            st.info(f" Using the best performing model based on ROC-AUC score")
            st.success(f" **Best Model:** {best_model} with ROC-AUC: {best_score:.4f}")
        else:
            st.info(" Using the best performing model based on ROC-AUC score")
        
        # Create layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Enter Student Information")
            
            # Create input form
            with st.form("prediction_form"):
                # Basic Information
                st.write("**📋 Basic Information**")
                age = st.slider("Age at enrollment", 16, 70, 20)
                gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
                admission_grade = st.slider("Admission grade", 95.0, 190.0, 130.0, step=0.1)
                
                # Academic Information - First Semester
                st.write("**📚 First Semester Performance**")
                curr_1st_enrolled = st.slider("Curricular units 1st sem (enrolled)", 0, 20, 6)
                curr_1st_approved = st.slider("Curricular units 1st sem (approved)", 0, 20, 5)
                curr_1st_grade = st.slider("Curricular units 1st sem (grade)", 0.0, 20.0, 12.0, step=0.1)
                curr_1st_evaluations = st.selectbox("Curricular units 1st sem (evaluations)", 
                                                   [0, 1], format_func=lambda x: "No evaluations" if x == 0 else "Has evaluations")
                
                # Academic Information - Second Semester
                st.write("**📖 Second Semester Performance**")
                curr_2nd_enrolled = st.slider("Curricular units 2nd sem (enrolled)", 0, 20, 6)
                curr_2nd_approved = st.slider("Curricular units 2nd sem (approved)", 0, 20, 5)
                curr_2nd_grade = st.slider("Curricular units 2nd sem (grade)", 0.0, 20.0, 12.0, step=0.1)
                curr_2nd_evaluations = st.selectbox("Curricular units 2nd sem (evaluations)",
                                                   [0, 1], format_func=lambda x: "No evaluations" if x == 0 else "Has evaluations")
                
                # Financial Information
                st.write("**💰 Financial Information**")
                scholarship = st.selectbox("Scholarship holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                debtor = st.selectbox("Debtor", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                tuition_up_to_date = st.selectbox("Tuition fees up to date", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
                
                # Family Background
                st.write("**👨‍👩‍👧‍👦 Family Background**")
                father_occupation = st.selectbox("Father's occupation", [0, 1], 
                                                format_func=lambda x: "No occupation/Unknown" if x == 0 else "Has occupation")
                mother_occupation = st.selectbox("Mother's occupation", [0, 1],
                                                format_func=lambda x: "No occupation/Unknown" if x == 0 else "Has occupation")
                
                # Socioeconomic Indicators
                st.write("**📊 Socioeconomic Context**")
                unemployment_rate = st.slider("Unemployment rate (%)", 7.0, 17.0, 10.0, step=0.1)
                inflation_rate = st.slider("Inflation rate (%)", -1.0, 4.0, 1.0, step=0.1)
                gdp = st.slider("GDP", -5.0, 5.0, 1.0, step=0.1)

                manual_inputs = {
                    "Age at enrollment": age,
                    "Gender": gender,
                    "Admission grade": admission_grade,
                    "Curricular units 1st sem (enrolled)": curr_1st_enrolled,
                    "Curricular units 1st sem (approved)": curr_1st_approved,
                    "Curricular units 1st sem (grade)": curr_1st_grade,
                    "Curricular units 1st sem (evaluations)": curr_1st_evaluations,
                    "Curricular units 2nd sem (enrolled)": curr_2nd_enrolled,
                    "Curricular units 2nd sem (approved)": curr_2nd_approved,
                    "Curricular units 2nd sem (grade)": curr_2nd_grade,
                    "Curricular units 2nd sem (evaluations)": curr_2nd_evaluations,
                    "Scholarship holder": scholarship,
                    "Debtor": debtor,
                    "Tuition fees up to date": tuition_up_to_date,
                    "Father's occupation": father_occupation,
                    "Mother's occupation": mother_occupation,
                    "Unemployment rate": unemployment_rate,
                    "Inflation rate": inflation_rate,
                    "GDP": gdp,
                }

                # ─── 3) Build the final feature_inputs by merging manual + dynamic ─────────
                feature_inputs = {}
                for feat in SELECTED_FEATURES:
                    if feat in manual_inputs:
                        # Use the widget value you already created
                        feature_inputs[feat] = manual_inputs[feat]
                    else:
                        # Fallback: dynamically generate widget based on data dtype
                        if pd.api.types.is_numeric_dtype(data[feat]):
                            default = float(data[feat].median())
                            feature_inputs[feat] = st.number_input(feat, value=default)
                        else:
                            opts = sorted(data[feat].dropna().unique().tolist())
                            feature_inputs[feat] = st.selectbox(feat, opts)
                    
                # Submit button
                submitted = st.form_submit_button("🔍 Predict Student Status", type="primary")
        
        with col2:
            st.subheader("Prediction Results")
            
            if submitted:
                try:
                    # Prepare input data - create DataFrame with all required features
                    if 'feature_columns' in components:
                        feature_cols = components['feature_columns']
                        input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
                    else:
                        # Fallback: use the original data columns (excluding Target)
                        if data is not None:
                            feature_cols = [col for col in data.columns if col != DEFAULT_TARGET]
                            input_data = pd.DataFrame(0, index=[0], columns=feature_cols)
                        else:
                            st.error("Cannot determine feature columns. Please check the model files.")
                            st.stop()
                    
                    input_data = data[SELECTED_FEATURES].copy()
                    feature_mapping = feature_inputs
                    
                    for feat, val in feature_inputs.items():
                        input_data[feat] = val
                    
                    # Make prediction
                    prediction = model.predict(input_data)
                    
                    # Get prediction probabilities if available
                    try:
                        prediction_proba = model.predict_proba(input_data)[0]
                    except:
                        prediction_proba = None
                    
                    # Decode prediction result
                    if 'class_names' in components:
                        class_names = components['class_names']
                        prediction_label = class_names[prediction[0]]
                    elif 'label_encoder' in components:
                        le = components['label_encoder']
                        # Label encoder uses inverse_transform but we need to map to class names
                        label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
                        prediction_label = label_map.get(prediction[0], 'Unknown')
                    else:
                        # Fallback mapping
                        label_map = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
                        prediction_label = label_map.get(prediction[0], 'Unknown')
                    
                    # Display prediction results
                    st.write("### 🎯 Predicted Status:")
                    
                    if prediction_label == "Dropout":
                        st.error("🚨 **Prediction: Likely to Drop Out**")
                        st.write("This student is at high risk of dropping out. Immediate intervention and support are recommended.")
                    elif prediction_label == "Graduate":
                        st.success("🎓 **Prediction: Likely to Graduate**")
                        st.write("This student shows strong indicators for successful graduation. Continue providing support.")
                    else:
                        st.info("📚 **Prediction: Likely to Stay Enrolled**")
                        st.write("This student is likely to continue studies but may need ongoing monitoring and support.")
                    
                    # Display prediction probabilities
                    if prediction_proba is not None:
                        st.write("Prediction Confidence:")
                        
                        # Get class names for probability display
                        if 'class_names' in components:
                            class_names = components['class_names']
                        else:
                            class_names = ['Dropout', 'Enrolled', 'Graduate']
                        
                        # Create probability chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
                        bars = ax.bar(class_names, prediction_proba, color=colors, alpha=0.8)
                        
                        ax.set_ylim(0, 1)
                        ax.set_ylabel('Probability')
                        ax.set_title('Prediction Confidence by Outcome')
                        
                        # Add percentage labels on bars
                        for i, (bar, prob) in enumerate(zip(bars, prediction_proba)):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{prob:.1%}', ha='center', va='bottom', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Show confidence level
                        max_prob = max(prediction_proba)
                        if max_prob > 0.8:
                            st.success(f" **High Confidence:** {max_prob:.1%}")
                        elif max_prob > 0.6:
                            st.info(f" **Medium Confidence:** {max_prob:.1%}")
                        else:
                            st.warning(f" **Low Confidence:** {max_prob:.1%}")
                    
                    # Show input summary
                    with st.expander(" Input Summary"):
                        summary_data = {
                            "Feature": list(feature_mapping.keys()),
                            "Value": list(feature_mapping.values())
                        }
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Error during prediction: {str(e)}")
                    st.write("Please check that all input data is valid and the model is properly trained.")
                    with st.expander("Debug Information"):
                        st.write(f"Model type: {type(model)}")
                        st.write(f"Available components: {list(components.keys())}")
            else:
                st.info("Please fill in the student information on the left and click 'Predict Student Status'.")

elif page == "Model Evaluation":
    st.header("📈 Model Performance Evaluation")
    
    if not model_loaded:
        st.warning("⚠️ Model not loaded. Please run the training script first.")
        st.stop()
    
    # Display performance metrics
    if 'results' in performance_data:
        st.subheader("📊 Performance Comparison")
        results_df = performance_data['results']
        
        # Display metrics in columns
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # Create performance visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        x = np.arange(len(results_df.index))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            if metric in results_df.columns:
                ax.bar(x + i*width, results_df[metric], width, label=metric.upper())
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels(results_df.index, rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Highlight best model
        best_model = results_df['roc_auc'].idxmax()
        best_score = results_df['roc_auc'].max()
        st.success(f"🏆 **Best Model:** {best_model} with ROC-AUC: {best_score:.4f}")
    
    # Show feature importance
    if 'feature_importance' in performance_data:
        st.subheader("🎯 Feature Importance Analysis")
        
        importance_df = performance_data['feature_importance']
        top_features = importance_df.head(15)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=top_features, x='importance', y='feature', ax=ax, palette='viridis')
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Features')
        ax.set_title('Top 15 Most Important Features')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show feature importance table
        with st.expander("📋 Complete Feature Importance Table"):
            st.dataframe(importance_df, use_container_width=True)

elif page == "Model Interpretation":
    st.header("🧠 Model Interpretation and Insights")
    
    st.subheader("🎯 Key Success Factors")
    
    # Create insights cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📚 Academic Performance
        - **First Semester Results**: Early academic performance strongly predicts final outcomes
        - **Course Completion Rate**: Higher approval rates indicate better study habits
        - **Grade Consistency**: Stable performance across semesters is crucial
        - **Evaluation Participation**: Active engagement in assessments matters
        """)
    
    with col2:
        st.markdown("""
        ### 💰 Financial Stability  
        - **Scholarship Impact**: Financial support significantly improves graduation rates
        - **Tuition Payments**: Timely payments indicate financial stability
        - **Debt Status**: Financial obligations can affect academic focus
        - **Economic Environment**: External economic factors influence student success
        """)
    
    with col3:
        st.markdown("""
        ### 👥 Personal & Social Factors
        - **Age at Enrollment**: Younger students may have different success patterns
        - **Family Background**: Parental occupation can influence educational outcomes
        - **Gender Differences**: May indicate different support needs
        - **Admission Preparation**: Initial academic readiness is important
        """)
    
    st.markdown("---")
    
    # Key insights section
    st.subheader("💡 Key Insights from Analysis")
    
    insights = [
        "**Early Warning System**: Students struggling in the first semester need immediate intervention",
        "**Financial Support Effectiveness**: Scholarships and financial aid significantly reduce dropout risk",
        "**Holistic Approach**: Multiple factors must be considered together for accurate predictions",
        "**Continuous Monitoring**: Regular assessment of student progress enables timely support",
        "**Personalized Interventions**: Different student profiles require tailored support strategies"
    ]
    
    for insight in insights:
        st.write(f"• {insight}")
    
    st.markdown("---")
    
    # Recommendations section
    st.subheader("🎯 Actionable Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🏫 For Educational Institutions
        - **Implement Early Warning Systems** to identify at-risk students
        - **Expand Financial Aid Programs** to support economically disadvantaged students  
        - **Develop Academic Support Services** including tutoring and mentoring
        - **Create Student Engagement Programs** to build community and belonging
        - **Regular Progress Monitoring** with personalized feedback and guidance
        - **Faculty Training** on identifying and supporting struggling students
        """)
    
    with col2:
        st.markdown("""
        ### 👨‍🎓 For Students
        - **Focus on First Semester Success** as it sets the foundation
        - **Actively Seek Academic Help** when facing difficulties
        - **Engage in Campus Activities** to build support networks
        - **Manage Finances Wisely** and explore financial aid options
        - **Develop Strong Study Habits** and time management skills
        - **Communicate with Faculty** about challenges and goals
        """)

elif page == "Student Profiles":
    st.header("👥 Student Profile Analysis")
    
    # Student type selection
    profile_type = st.selectbox(
        "Select Student Profile Type",
        ["High Dropout Risk", "Likely to Graduate", "Enrolled Students Needing Support"]
    )
    
    if profile_type == "High Dropout Risk":
        st.subheader("🚨 High Dropout Risk Student Profile")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### 📊 Risk Indicators
            - Few courses passed in first semester (< 50%)
            - Low average grades (< 11/20)
            - Frequent tuition payment delays
            - Limited campus engagement
            - No scholarship support
            - High absence rates
            """)
        
        with col2:
            st.markdown("""
            ### ⚠️ Warning Signs
            - **Academic**: Failing multiple courses, missing assignments
            - **Financial**: Difficulty paying tuition, expressing financial stress
            - **Social**: Limited interaction with peers and faculty
            - **Behavioral**: Declining attendance, disengagement from activities
            - **Personal**: Expressing doubt about academic goals
            """)
        
        st.subheader("🎯 Intervention Strategies")
        strategies = [
            "**Immediate Academic Support**: Assign tutors and provide study resources",
            "**Financial Assistance**: Connect with financial aid and emergency funds",
            "**Personal Counseling**: Provide access to mental health and academic advisors",
            "**Skill Development**: Offer workshops on study techniques and time management",
            "**Peer Support**: Create mentoring relationships with successful students",
            "**Flexible Options**: Consider part-time enrollment or course load reduction"
        ]
        
        for strategy in strategies:
            st.write(f"• {strategy}")
            
    elif profile_type == "Likely to Graduate":
        st.subheader("🎓 High Success Probability Student Profile")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### ✅ Success Indicators
            - High course completion rate (> 80%)
            - Strong academic performance (> 13/20)
            - Consistent tuition payments
            - Active campus participation
            - Often have scholarships
            - Good attendance records
            """)
        
        with col2:
            st.markdown("""
            ### 🌟 Success Behaviors
            - **Academic**: Proactive help-seeking, consistent study habits
            - **Financial**: Good financial planning and management
            - **Social**: Strong peer networks and faculty relationships
            - **Goal-oriented**: Clear academic and career objectives
            - **Resilient**: Ability to overcome challenges effectively
            """)
        
        st.subheader("🚀 Enhancement Strategies")
        strategies = [
            "**Advanced Opportunities**: Offer research projects and internships",
            "**Leadership Development**: Encourage roles in student organizations",
            "**Career Preparation**: Provide career counseling and job placement services",
            "**Peer Mentoring**: Train as mentors for struggling students",
            "**Graduate School Preparation**: Offer guidance on advanced education",
            "**Recognition Programs**: Acknowledge and celebrate achievements"
        ]
        
        for strategy in strategies:
            st.write(f"• {strategy}")
            
    else:  # Enrolled Students Needing Support
        st.subheader("📚 Enrolled Students Requiring Attention")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### 📈 Mixed Indicators
            - Moderate performance (11-13/20)
            - Some course retakes needed
            - Occasional payment issues
            - Variable engagement levels
            - May face personal challenges
            - Inconsistent attendance
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Attention Signals
            - **Academic**: Fluctuating grades, difficulty in specific subjects
            - **Financial**: Occasional payment delays or budget concerns
            - **Social**: Limited but present campus connections
            - **Motivation**: Periods of high and low engagement
            - **Personal**: Balancing multiple life responsibilities
            """)
        
        st.subheader("🤝 Support Strategies")
        strategies = [
            "**Regular Check-ins**: Schedule periodic meetings with academic advisors",
            "**Targeted Subject Support**: Provide specialized tutoring for challenging courses",
            "**Peer Study Groups**: Facilitate collaborative learning opportunities",
            "**Flexible Learning Options**: Offer alternative scheduling and formats",
            "**Life Skills Training**: Provide time management and stress reduction workshops",
            "**Resource Connections**: Link to campus resources and support services"
        ]
        
        for strategy in strategies:
            st.write(f"• {strategy}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Student Performance Prediction System</strong></p>
    <p>Developed by Group 3 | Early Prediction of Higher Education Student Performance Using Machine Learning</p>
    <p>Powered by Streamlit • Built with for Educational Success</p>
</div>
""", unsafe_allow_html=True)