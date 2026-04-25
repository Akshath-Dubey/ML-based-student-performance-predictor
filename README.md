# 🎓 Student Performance Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ml-based-student-performance-predictor-3ppgjmaoyqwulr3kmm3dapp.streamlit.app/)

An interactive ML-powered dashboard that predicts student academic performance based on behavioral features like classroom participation, resource usage, and attendance.

## 🚀 Live Demo

**Try it here:** [https://ml-based-student-performance-predictor-3ppgjmaoyqwulr3kmm3dapp.streamlit.app/](https://ml-based-student-performance-predictor-3ppgjmaoyqwulr3kmm3dapp.streamlit.app/)

## ✨ Features

- **9 Interactive Visualizations** — Explore data through charts (Class distribution, Gender-wise, Nationality-wise, Grade-wise, etc.)
- **ML Model Comparison** — Train and compare 5 algorithms:
  - Decision Tree
  - Random Forest
  - Perceptron
  - Logistic Regression
  - Multi-Layer Perceptron (MLP)
- **Live Prediction** — Input student metrics and get instant performance predictions with recommendations
- **Majority Voting** — Ensemble prediction from all models for robust results

## 🛠️ Tech Stack

| Technology               | Purpose                 |
| ------------------------ | ----------------------- |
| **Streamlit**            | Web dashboard framework |
| **Python**               | Backend logic & ML      |
| **Pandas**               | Data manipulation       |
| **Seaborn & Matplotlib** | Static visualizations   |
| **Plotly**               | Interactive charts      |
| **Scikit-learn**         | Machine learning models |

## 📊 Dataset

**File:** `AI-Data.csv`

| Feature              | Description                                      |
| -------------------- | ------------------------------------------------ |
| `raisedhands`        | Number of times student raised hands (0-100)     |
| `VisITedResources`   | Resources visited online (0-100)                 |
| `Discussion`         | Group discussion participation (0-100)           |
| `StudentAbsenceDays` | Attendance: Under-7 or Above-7 days              |
| **Target: `Class`**  | Performance level: L (Low), M (Medium), H (High) |

## 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Akshath-Dubey/ML-based-student-performance-predictor.git

# Navigate to project
cd ML-based-student-performance-predictor

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**Local URL:** http://localhost:8501

## 📁 Project Structure

```
.
├── app.py              # Main Streamlit application
├── AI-Data.csv         # Dataset
├── requirements.txt    # Python dependencies
└── .gitignore          # Excluded files
```

## 🤖 Machine Learning Pipeline

1. **Data Preprocessing**
   - Label encoding for categorical features
   - Train-test split (70-30) with stratification

2. **Model Training**
   - 5 classifiers trained on behavioral features
   - Accuracy evaluated on test set

3. **Prediction**
   - User inputs student metrics
   - All 5 models predict individually
   - Majority vote determines final result
   - Personalized recommendation provided

## 📈 Model Accuracies

Models are automatically trained and their accuracies displayed in both table and bar chart format within the app.

## 🎯 Use Cases

- Teachers identifying at-risk students early
- Parents understanding factors affecting performance
- Students self-assessing study habits
- Educational institutions analyzing trends

---

**Deployed on:** [Streamlit Community Cloud](https://streamlit.io/cloud)  
**Developer:** [Akshath Dubey](https://github.com/Akshath-Dubey)
