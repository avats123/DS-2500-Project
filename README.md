# 🌙 Sleep Health & Lifestyle: A Multi-Model Analysis
**DS 2500 Project | Exploratory Data Analysis & Predictive Modeling**

## 📊 Project Objective
The goal of this project is to effectively analyze a comprehensive sleep dataset using a variety of Data Science techniques. By leveraging **Exploratory Data Analysis (EDA)**, **Machine Learning (kNN)**, and **Linear/Logistic Regression**, this project investigates the complex relationships between lifestyle factors (screen time, caffeine, physical activity) and health metrics (BMI, heart rate) to determine their impact on sleep quality and disorder risks.

## 🗂️ Repository Structure
This project utilizes a feature-branching workflow to ensure code stability and organized development.

* **`main`**: The primary branch containing stable, peer-reviewed scripts and final visualizations.
* **`ansh-EDA`**: The development branch used for feature engineering, correlation studies, and initial model prototyping.

---

## 🧬 Dataset & Features
The analysis is performed on a 100,000-row dataset covering 32 distinct features, providing a robust foundation for statistical significance.

### Primary Variables of Interest:
* **Lifestyle Factors:** Screen time (mins), caffeine intake (mg), alcohol consumption, and smoking status.
* **Physical Metrics:** BMI, resting heart rate, and daily step counts.
* **Sleep Outcomes:** Sleep duration, quality scores, REM/Deep sleep percentages, and latency.
* **Categorical Groups:** Occupation, gender, and chronotype (Morning Lark vs. Night Owl).

---

## 🔍 Data Science Methodology

### 1. Exploratory Data Analysis (EDA)
We utilize EDA to uncover hidden patterns and validate hypotheses regarding sleep hygiene. 
* **Correlation Analysis:** Investigating how screen time and caffeine before bed directly influence sleep latency.
* **Occupational Trends:** Comparing stress levels and sleep duration across different career paths.
* **Categorical Binning:** Grouping activity levels and age brackets to find non-linear trends in sleep quality.

### 2. Predictive Modeling (kNN)
A **k-Nearest Neighbors** classifier is implemented to categorize individuals into sleep disorder risk levels: *Healthy, Mild, Moderate, or Severe*.
* **Preprocessing:** Features are scaled using `StandardScaler` and categorical variables are transformed via One-Hot Encoding.
* **Evaluation:** Using classification reports and confusion matrices to assess model accuracy and recall across all risk levels.

### 3. Regression & Statistical Analysis
We employ **Linear and Logistic Regression** to quantify the influence of specific predictors on sleep outcomes.
* **Linear Regression:** Used to model continuous outcomes like `sleep_duration_hrs` based on physical activity and stress scores.
* **Logistic Regression:** Used to calculate the probability of "Felt Rested" based on sleep architecture (REM/Deep sleep percentages).

---

## 🚀 How to Run

1. **Clone and Setup:**
   ```bash
   # Clone the repository
   git clone [https://github.com/](https://github.com/)[your-username]/DS-2500-Project.git
   cd DS-2500-Project

   # Set up the virtual environment
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

   # Install dependencies from requirements.txt
   pip install -r requirements.txt

   # Execute the analysis
   python scripts/data_loading.py
