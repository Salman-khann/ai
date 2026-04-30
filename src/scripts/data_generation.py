import pandas as pd
from ctgan import CTGAN

def generate_digital_twin():
    print("Loading seed dataset...")
    
    # Define Base Directory (Project Root)
    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(BASE_DIR, "src", "data", "processed", "oulad_seed.csv")
    output_path = os.path.join(BASE_DIR, "src", "data", "processed", "synthetic_students.csv")

    try:
        real_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print("Error: Could not find oulad_seed.csv. Make sure you are running this script in the datasets folder.")
        return

    print("Cleaning data: Removing rows with missing values...")
    real_data.dropna(inplace=True)

    if 'id_student' in real_data.columns:
        print("Dropping 'id_student' column before training...")
        real_data.drop(columns=['id_student'], inplace=True)

    # --- THE BULLETPROOF FIX ---
    # Tell pandas to automatically find every column that contains text (objects)
    # and create our discrete_columns list for us!
    discrete_columns = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
    print(f"Automatically detected text columns for the AI: {discrete_columns}")
    # ---------------------------

    print(f"Training CTGAN Model on {len(real_data)} clean records (This may take 5-10 minutes)...")
    
    ctgan = CTGAN(epochs=150) 
    ctgan.fit(real_data, discrete_columns)

    print("Generating 5,000 synthetic student profiles...")
    synthetic_data = ctgan.sample(5000)

    synthetic_data.to_csv(output_path, index=False)
    
    print(f"Success! 5,000 synthetic profiles saved to {output_path}")

if __name__ == '__main__':
    generate_digital_twin()