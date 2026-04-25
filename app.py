import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="Student Performance Predictor", layout="wide", page_icon="📊")

# Load and prepare data (shared)
@st.cache_data
def load_raw_data():
    raw_data = pd.read_csv("AI-Data.csv").fillna("")
    return raw_data

@st.cache_data
def load_and_prepare_data():
    data = pd.read_csv("AI-Data.csv")
    drop_cols = ["gender", "StageID", "GradeID", "NationalITy", "PlaceofBirth", "SectionID", 
                 "Topic", "Semester", "Relation", "ParentschoolSatisfaction", "ParentAnsweringSurvey", "AnnouncementsView"]
    data_ml = data.drop(columns=[col for col in drop_cols if col in data.columns]).fillna(0)
    
    feature_cols = ['raisedhands', 'VisITedResources', 'Discussion', 'StudentAbsenceDays']
    X = data_ml[feature_cols].copy()
    y = data_ml['Class'].copy()
    
    le_X = LabelEncoder()
    X['StudentAbsenceDays'] = le_X.fit_transform(X['StudentAbsenceDays'])
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    return data_ml, X, y, X_train, X_test, y_train, y_test, le_y, le_X

# Train models and get accuracies
@st.cache_data
def train_models(X_train, y_train, X_test, y_test):
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Perceptron': Perceptron(random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'MLP': MLPClassifier(activation='logistic', random_state=42, max_iter=1000)
    }
    accuracies = {}
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        accuracies[name] = acc
        trained_models[name] = model
    return accuracies, trained_models

# Create seaborn plot as plotly fig
def seaborn_to_plotly(fig):
    return fig

# Graph generation functions
def plot_graph1(data):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(x='Class', data=data, order=['L', 'M', 'H'], ax=ax)
    ax.set_title('1. Marks Class Count')
    st.pyplot(fig)

def plot_graph2(data):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='Semester', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('2. Semester-wise')
    st.pyplot(fig)

def plot_graph3(data):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='gender', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('3. Gender-wise')
    st.pyplot(fig)

def plot_graph4(data):
    fig, ax = plt.subplots(figsize=(14,6))
    sns.countplot(x='NationalITy', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('4. Nationality-wise')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_graph5(data):
    fig, ax = plt.subplots(figsize=(14,6))
    sns.countplot(x='GradeID', hue='Class', data=data, 
                  order=['G-02', 'G-04', 'G-05', 'G-06', 'G-07', 'G-08', 'G-09', 'G-10', 'G-11', 'G-12'], 
                  hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('5. Grade-wise')
    st.pyplot(fig)

def plot_graph6(data):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(x='SectionID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('6. Section-wise')
    st.pyplot(fig)

def plot_graph7(data):
    fig, ax = plt.subplots(figsize=(14,6))
    sns.countplot(x='Topic', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('7. Topic-wise')
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_graph8(data):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.countplot(x='StageID', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('8. Stage-wise')
    st.pyplot(fig)

def plot_graph9(data):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.countplot(x='StudentAbsenceDays', hue='Class', data=data, hue_order=['L', 'M', 'H'], ax=ax)
    ax.set_title('9. Absent Days-wise')
    st.pyplot(fig)

# Main app
st.title('📊 Student Performance Predictor - Web Dashboard')
st.markdown('---')

raw_data = load_raw_data()
data_ml, X, y, X_train, X_test, y_train, y_test, le_y, le_X = load_and_prepare_data()
accuracies, trained_models = train_models(X_train, y_train, X_test, y_test)

# Columns layout
col1, col2 = st.columns([2,1])

with col1:
    st.header('🔍 Interactive Visualizations')
    st.markdown('Click buttons to generate all 9 analysis graphs:')
    
    if st.button('1. Marks Class Count', width='stretch'):
        plot_graph1(raw_data)
    if st.button('2. Semester-wise', width='stretch'):
        plot_graph2(raw_data)
    if st.button('3. Gender-wise', width='stretch'):
        plot_graph3(raw_data)
    if st.button('4. Nationality-wise', width='stretch'):
        plot_graph4(raw_data)
    if st.button('5. Grade-wise', width='stretch'):
        plot_graph5(raw_data)
    if st.button('6. Section-wise', width='stretch'):
        plot_graph6(raw_data)
    if st.button('7. Topic-wise', width='stretch'):
        plot_graph7(raw_data)
    if st.button('8. Stage-wise', width='stretch'):
        plot_graph8(raw_data)
    if st.button('9. Absent Days-wise', width='stretch'):
        plot_graph9(raw_data)

with col2:
    st.header('📈 ML Model Accuracies')
    
    # Table
    acc_df = pd.DataFrame(list(accuracies.items()), columns=['Model', 'Accuracy'])
    st.dataframe(acc_df.style.format({'Accuracy': '{:.4f}'}), width='stretch')
    
    # Bar chart
    fig_acc = px.bar(acc_df, x='Model', y='Accuracy', 
                     title='Model Accuracy Comparison',
                     color='Accuracy', color_continuous_scale='viridis')
    fig_acc.update_layout(height=300, xaxis_tickangle=45)
    st.plotly_chart(fig_acc, width='stretch')

# Features section
st.header('🔧 Prediction Features')
st.info(f"""
**Used for ML Predictions:**
- **raisedhands** (0-100): Times student raised hands
- **VisITedResources** (0-100): Resources viewed online
- **Discussion** (0-100): Group discussion participation
- **StudentAbsenceDays**: Under-7=1, Above-7=0

**Target:** Class (L=Low, M=Medium, H=High)
""")

# Prediction form
st.header('🧪 Test Specific Student Prediction')
with st.form('prediction_form'):
    col_p1, col_p2, col_p3, col_p4 = st.columns(4)
    with col_p1: raisedhands = st.number_input('Raised Hands (0-100)', 0, 100, 20)
    with col_p2: visited = st.number_input('Visited Resources (0-100)', 0, 100, 20)
    with col_p3: discussion = st.number_input('Discussion (0-100)', 0, 100, 25)
    with col_p4: absence = st.selectbox('Absence Days', ['Under-7 (1)', 'Above-7 (0)'], index=0)
    absence_encoded = 1 if absence == 'Under-7 (1)' else 0
    
    submit = st.form_submit_button('Predict Performance', width='stretch')
    
    if submit:
        test_sample = np.array([[raisedhands, visited, discussion, absence_encoded]])
        preds = {}
        for name, model in trained_models.items():
            pred_encoded = model.predict(test_sample)[0]
            pred_label = le_y.inverse_transform([pred_encoded])[0]
            preds[name] = pred_label
        
        st.subheader('Predictions:')
        pred_df = pd.DataFrame(list(preds.items()), columns=['Model', 'Prediction'])
        st.dataframe(pred_df, width='stretch')
        
        # Majority vote
        votes_encoded = [le_y.transform([preds[name]])[0] for name in preds]
        majority_encoded = np.bincount(votes_encoded).argmax()
        final_pred = le_y.inverse_transform([majority_encoded])[0]
        st.success(f'**Final Prediction (Majority Vote): {final_pred}**')
        
        # Recommendation
        recs = {
            'L': 'Increase attendance, participate in discussions, review material regularly.',
            'M': 'Performance average. Maintain habits, focus on weak areas.',
            'H': 'Excellent! Keep up the outstanding performance.'
        }
        st.balloons()
        st.markdown(f'**Recommendation:** {recs[final_pred]}')
