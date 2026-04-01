import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# 1. Use os.path.join to handle folder paths correctly
# This looks into the 'data' folder for your file
file_path = os.path.join('data', 'sleep_health_dataset.csv')

# Begin cleaning/scaling data for statistics/EDA
def clean_and_prep_data():
    df = pd.read_csv(file_path)

    # Compute summary statistics and learn about structure of dataset
    print(df.info()) 
    print(df.describe())

    df = df.drop_duplicates()
    # Note: Your dataset is already pretty clean, but this prevents errors later!
    
    # 3. Create Age Groups (4 Bins)
    # Using 'labels' makes your charts much easier to read later
    df['age_group'] = pd.cut(df['age'], bins=4, 
                             labels=['Young Adult (18-30)', 'Early Mid-Life (31-43)', 
                                     'Late Mid-Life (44-56)', 'Senior (57-69)'])
    
    # 4. Create Step Groups (4 Bins)
    df['step_category'] = pd.cut(df['steps_that_day'], bins=4, 
                                  labels=['Low Activity', 'Moderate Activity', 
                                          'High Activity', 'Extreme Activity'])
    
    # 5. Clean Categorical Text (Optional but helpful)
    # Ensure all occupations are capitalized consistently
    if 'occupation' in df.columns:
        df['occupation'] = df['occupation'].str.strip().str.title()

    return df


# Group by Occupation and compute average sleep score and average 
def get_occupation_statistics(df):
    occ_comparison = df.groupby('occupation')[['sleep_duration_hrs', 'stress_score']].mean()
    occ_comparison = occ_comparison.sort_values(by='sleep_duration_hrs')

    print("--- Occupation Sleep Comparison ---")
    print(occ_comparison)   

# Creating a grouped steps analysis function which groups steps into four categories and computers
# the average sleep quality, duration, and deep sleep percentage 
def analyze_sleep_by_steps(df):
    """
    Groups data into 4 activity tiers and calculates 
    average quality, duration, and deep sleep.
    """
    # List of metrics to compute
    metrics = ['sleep_quality_score', 'sleep_duration_hrs', 'deep_sleep_percentage']
    
    # Group and calculate means
    step_summary = df.groupby('step_category', observed=True)[metrics].mean()
    return step_summary

# Creating a grouped steps analysis function which people into whether they exercise or not
# the average sleep quality, duration, and deep sleep percentage 
def analyze_sleep_by_exercise_status(df):
    """
    Compares core sleep metrics between those who 
    exercised (1) and those who did not (0).
    """
    metrics = ['sleep_quality_score', 'sleep_duration_hrs', 'deep_sleep_percentage']
    
    # Group by exercise_day and calculate means
    exercise_summary = df.groupby('exercise_day')[metrics].mean()
    
    # Rename indexes to No Exercise, Exercise
    exercise_summary.index = ['No Exercise', 'Exercised']
    return exercise_summary

def analyze_sleep_by_age(df):
    """
    Compares the core sleep metrics between the different
    age groups assigned in the first function
    """
    
    metrics = ['sleep_quality_score', 'sleep_duration_hrs', 'deep_sleep_percentage']

    # Group by age group and calculate means
    age_summary = df.groupby('age_group')[metrics].mean()
    return age_summary

def analyze_bedtime_habits(df):
    # Correlation between screen time, caffeine, and latency
    # 1. Force these columns to be numeric (just in case they loaded as strings)
    cols_to_fix = ['screen_time_before_bed_mins', 'caffeine_mg_before_bed']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 2. Calculate Correlation using ONLY these numeric columns
    # We create a mini-dataframe first so .corr() doesn't get distracted
    bedtime_df = df[cols_to_fix].dropna()
    correlation = bedtime_df.corr()
    
    print("--- Bedtime Habit Correlations ---")
    # This prevents the KeyError by checking if it exists in the result
    if 'sleep_latency_mins' in correlation.columns:
        print(correlation['sleep_latency_mins'])
    else:
        print("Error: sleep_latency_mins was dropped. Check for non-numeric data.")
    
    # Or group screen time into 30-min blocks
    df['screen_time_bins'] = pd.cut(df['screen_time_before_bed_mins'], bins=5)
    return df.groupby('screen_time_bins', observed=True)['sleep_latency_mins'].mean()

def run_final_knn_model(df):
    print("\n--- Running kNN: Predicting Sleep Disorder Risk ---")
    
    # 1. Handle Categorical Data (One-Hot Encoding)
    # This creates the dummy columns for Occupation, BMI, and Chronotype
    df_encoded = pd.get_dummies(df, columns=['occupation', 'bmi', 'chronotype'], drop_first=True)
    
    # 2. Separate Features (X) and Target (y)
    # We drop the Target and any non-predictive columns (ID, Bins)
    X = df_encoded.select_dtypes(include=['number', 'bool'])
    
    cols_to_remove = ['person_id', 'sleep_disorder_risk'] 
    X = X.drop(columns=[c for c in cols_to_remove if c in X.columns])

    y = df_encoded['sleep_disorder_risk']
    
    # 3. Create Training (80%) and Testing (20%) Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Scale the Data
    # Stress (1-10) and Steps (5000+) must be on the same scale for kNN math
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Initialize and Train the Model
    # k=9 is a good balance for a dataset of 100k rows
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)
    
    # 6. Make Predictions
    y_pred = knn.predict(X_test_scaled)
    
    # 7. Evaluate Performance
    score = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {score:.2%}")
    print("\nDetailed Performance Report:")
    print(classification_report(y_test, y_pred))

    # Return the objects in case you want to plot them later
    return y_test, y_pred

def main():
    df = clean_and_prep_data()
    get_occupation_statistics(df)

    print("--- Analysis by Daily Step Count ---")
    print(analyze_sleep_by_steps(df))

    print("\n--- Analysis by Exercise Participation ---")
    print(analyze_sleep_by_exercise_status(df))

    print("\n--- Analysis by Age Group ---")
    print(analyze_sleep_by_age(df))

    print("\n--- Analyzing Bedtime Habits (Screen Time, Caffeine Usage on Sleep Latency) ---")
    print(analyze_bedtime_habits(df))

    run_final_knn_model(df)

if __name__ == '__main__':
    main()