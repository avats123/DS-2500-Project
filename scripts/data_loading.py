import pandas as pd
import os

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


def main():
    df = clean_and_prep_data()
    get_occupation_statistics(df)

    print("--- Analysis by Daily Step Count ---")
    print(analyze_sleep_by_steps(df))

    print("\n--- Analysis by Exercise Participation ---")
    print(analyze_sleep_by_exercise_status(df))

    print("\n--- Analysis by Age Group ---")
    print(analyze_sleep_by_age(df))

if __name__ == '__main__':
    main()