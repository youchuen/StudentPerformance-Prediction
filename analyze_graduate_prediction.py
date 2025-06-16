import pandas as pd
import joblib
import numpy as np

# Load model and data
model = joblib.load('outputs/models/best_model.pkl')
data = pd.read_csv('data/processed/cleaned.csv')
feature_cols = joblib.load('outputs/models/feature_columns.pkl')
class_names = joblib.load('outputs/models/class_names.pkl')

print('=== ANALYSIS: HOW TO GET GRADUATE PREDICTION ===')

# Analyze graduate student characteristics
graduates = data[data['Target'] == 'Graduate']
dropouts = data[data['Target'] == 'Dropout']
enrolled = data[data['Target'] == 'Enrolled']

print(f'\nGraduate students (n={len(graduates)}):')
print(f'Mean 1st sem approved: {graduates["Curricular units 1st sem (approved)"].mean():.1f}')
print(f'Mean 2nd sem approved: {graduates["Curricular units 2nd sem (approved)"].mean():.1f}')
print(f'Mean 1st sem grade: {graduates["Curricular units 1st sem (grade)"].mean():.1f}')
print(f'Mean 2nd sem grade: {graduates["Curricular units 2nd sem (grade)"].mean():.1f}')
print(f'Tuition up to date rate: {graduates["Tuition fees up to date"].mean():.1%}')
print(f'Mean age: {graduates["Age at enrollment"].mean():.1f}')
print(f'Mean admission grade: {graduates["Admission grade"].mean():.1f}')

print('\n=== TESTING SAMPLE GRADUATE STUDENTS ===')
# Test actual graduate students to see model confidence
sample_graduates = graduates.sample(n=10, random_state=42)

high_confidence_graduates = []
for idx in sample_graduates.index:
    student_data = data.loc[[idx]]
    X_sample = student_data[feature_cols]
    
    try:
        proba = model.predict_proba(X_sample)[0]
        pred = model.predict(X_sample)[0]
        predicted_class = class_names[pred]
        
        if predicted_class == 'Graduate' and proba[2] > 0.7:  # High confidence graduates
            high_confidence_graduates.append({
                'index': idx,
                'graduate_prob': proba[2],
                'confidence': max(proba),
                'data': student_data
            })
    except Exception as e:
        print(f'Error with index {idx}: {e}')

print(f'\nFound {len(high_confidence_graduates)} high-confidence graduate predictions')

# Show top 3 high-confidence graduates
print('\n=== TOP HIGH-CONFIDENCE GRADUATE PROFILES ===')
sorted_graduates = sorted(high_confidence_graduates, key=lambda x: x['graduate_prob'], reverse=True)

for i, grad in enumerate(sorted_graduates[:3]):
    student = grad['data'].iloc[0]
    print(f'\n--- High-Confidence Graduate {i+1} ---')
    print(f'Graduate Probability: {grad["graduate_prob"]:.1%}')
    print(f'Overall Confidence: {grad["confidence"]:.1%}')
    print(f'Key characteristics:')
    print(f'  • Age: {student["Age at enrollment"]:.0f}')
    print(f'  • Admission grade: {student["Admission grade"]:.1f}')
    print(f'  • 1st sem approved: {student["Curricular units 1st sem (approved)"]:.0f}')
    print(f'  • 2nd sem approved: {student["Curricular units 2nd sem (approved)"]:.0f}')
    print(f'  • 1st sem grade: {student["Curricular units 1st sem (grade)"]:.1f}')
    print(f'  • 2nd sem grade: {student["Curricular units 2nd sem (grade)"]:.1f}')
    print(f'  • Tuition up to date: {student["Tuition fees up to date"]:.0f}')
    print(f'  • Scholarship: {student.get("Scholarship holder", "N/A")}')

print('\n=== IDEAL GRADUATE PROFILE RECIPE ===')
print('To get "Likely to Graduate" prediction, set these parameters:')
print()
print('📚 ACADEMIC EXCELLENCE:')
print('  • Curricular units 1st sem (approved): 6 (maximum)')
print('  • Curricular units 2nd sem (approved): 6 (maximum)')  
print('  • Curricular units 1st sem (grade): 13-15+ (out of 20)')
print('  • Curricular units 2nd sem (grade): 13-15+ (out of 20)')
print('  • Curricular units 1st sem (evaluations): 1 (has evaluations)')
print('  • Curricular units 2nd sem (evaluations): 1 (has evaluations)')

print('\n💰 FINANCIAL STABILITY:')
print('  • Tuition fees up to date: 1 (Yes)')
print('  • Debtor: 0 (No)')
print('  • Scholarship holder: 1 (Yes) - if available')

print('\n👤 DEMOGRAPHIC FACTORS:')
print('  • Age at enrollment: 18-22 (younger students)')
print('  • Admission grade: 130-150+ (higher admission scores)')
print('  • Gender: 0 or 1 (both work, slight variations)')

print('\n🏫 FAMILY/SOCIAL:')
print('  • Father\'s occupation: 1 (has occupation)')
print('  • Mother\'s occupation: 1 (has occupation)')

print('\n📊 ECONOMIC CONTEXT (use current values):')
print('  • Unemployment rate: 7-12% (varies by year)')
print('  • Inflation rate: -1 to 3% (varies by year)')
print('  • GDP: -5 to 5% (varies by year)')

print('\n=== EXAMPLE INPUT FOR GRADUATE PREDICTION ===')
print('Try these settings in your Streamlit app:')
print('Age: 19')
print('Admission grade: 140')
print('Gender: Female (0) or Male (1)')
print('1st sem enrolled: 6')
print('1st sem approved: 6') 
print('1st sem grade: 14')
print('1st sem evaluations: Has evaluations')
print('2nd sem enrolled: 6')
print('2nd sem approved: 6')
print('2nd sem grade: 14')
print('2nd sem evaluations: Has evaluations')
print('Scholarship: Yes')
print('Debtor: No')
print('Tuition up to date: Yes')
print('Father\'s occupation: Has occupation')
print('Mother\'s occupation: Has occupation')
print('Unemployment rate: 10%')
print('Inflation rate: 1%')
print('GDP: 1%')

print('\n🎯 This combination should give you "Likely to Graduate" with high confidence!')

# Test the ideal profile
print('\n=== TESTING IDEAL PROFILE ===')
ideal_profile = pd.DataFrame({
    'Age at enrollment': [19],
    'Admission grade': [140.0],
    'Gender': [0],
    'Curricular units 1st sem (enrolled)': [6],
    'Curricular units 1st sem (approved)': [6],
    'Curricular units 1st sem (grade)': [14.0],
    'Curricular units 1st sem (evaluations)': [1],
    'Curricular units 2nd sem (enrolled)': [6],
    'Curricular units 2nd sem (approved)': [6],
    'Curricular units 2nd sem (grade)': [14.0],
    'Curricular units 2nd sem (evaluations)': [1],
    'Scholarship holder': [1],
    'Debtor': [0],
    'Tuition fees up to date': [1],
    "Father's occupation": [1],
    "Mother's occupation": [1],
    'Unemployment rate': [10.0],
    'Inflation rate': [1.0],
    'GDP': [1.0]
})

# Add missing columns with defaults
for col in feature_cols:
    if col not in ideal_profile.columns:
        ideal_profile[col] = 0

# Reorder columns to match model
ideal_profile = ideal_profile[feature_cols]

try:
    ideal_proba = model.predict_proba(ideal_profile)[0]
    ideal_pred = model.predict(ideal_profile)[0]
    ideal_class = class_names[ideal_pred]
    
    print(f'Ideal Profile Prediction: {ideal_class}')
    print(f'Probabilities: Dropout={ideal_proba[0]:.1%}, Enrolled={ideal_proba[1]:.1%}, Graduate={ideal_proba[2]:.1%}')
    print(f'Confidence: {max(ideal_proba):.1%}')
    
    if ideal_class == 'Graduate':
        print('✅ SUCCESS! This profile predicts "Likely to Graduate"')
    else:
        print('❌ This profile needs adjustment')
        
except Exception as e:
    print(f'Error testing ideal profile: {e}') 