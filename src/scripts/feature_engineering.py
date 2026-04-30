# src/feature_engineering.py
import pandas as pd

def build_seed_dataset():
    print("Loading OULAD files...")
    # Update these paths to point to where you unzipped the 7 Kaggle CSVs
    try:
        info_df = pd.read_csv("C:\\Users\\salman\\Desktop\\ai\\src\\data\\raw\\studentInfo.csv")
        vle_df = pd.read_csv("C:\\Users\\salman\\Desktop\\ai\\src\\data\\raw\\studentVle.csv")
        assessment_df = pd.read_csv("C:\\Users\\salman\\Desktop\\ai\\src\\data\\raw\\studentAssessment.csv")
    except FileNotFoundError:
        print("Error: Could not find the Kaggle CSVs. Please check the folder paths.")
        return

    print("1. Processing Engagement Data (LMS Clicks)...")
    # Group the millions of clicks by student ID and sum them up
    student_clicks = vle_df.groupby('id_student')['sum_click'].sum().reset_index()
    student_clicks.rename(columns={'sum_click': 'total_lms_clicks'}, inplace=True)

    print("2. Processing Academic Data (Scores & Latency)...")
    student_stats = assessment_df.groupby('id_student').agg({
        'score': 'mean',
        'date_submitted': 'mean'
    }).reset_index()
    student_stats.rename(columns={'score': 'average_score', 'date_submitted': 'avg_submission_day'}, inplace=True)
    assessments_completed = assessment_df.groupby('id_student').size().reset_index(name='assessments_completed')

    print("3. Merging into the Master Seed File...")
    seed_df = info_df.copy()
    seed_df = pd.merge(seed_df, student_clicks, on='id_student', how='left')
    seed_df = pd.merge(seed_df, student_stats, on='id_student', how='left')
    seed_df = pd.merge(seed_df, assessments_completed, on='id_student', how='left')

    seed_df['imd_band'] = seed_df['imd_band'].fillna('Missing')
    seed_df.fillna({
        'total_lms_clicks': 0, 
        'average_score': 0, 
        'avg_submission_day': 200, 
        'assessments_completed': 0
    }, inplace=True)

    final_columns = [
        'id_student', 'gender', 'region', 'highest_education', 'imd_band', 
        'age_band', 'num_of_prev_attempts', 'studied_credits', 'disability', 
        'total_lms_clicks', 'average_score', 'avg_submission_day', 
        'assessments_completed', 'final_result'
    ]
    final_seed = seed_df[final_columns]

    print(f"Final Seed Dataset Shape: {final_seed.shape}")
    
    # Save the flattened file for the CTGAN to use
    output_path = "C:\\Users\\salman\\Desktop\\ai\\src\\data\\processed\\oulad_seed.csv"
    final_seed.to_csv(output_path, index=False)
    print(f"Success! Seed dataset saved to {output_path}")

if __name__ == '__main__':
    build_seed_dataset()