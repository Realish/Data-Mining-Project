import os
import pandas as pd
import librosa

# Path to the folder containing audio files
folder_path = '/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/Transformed_Data/Compression_(with_-20_db_threshold_and_2_to_1_ratio)/LCompressed_Data'

# Duration to consider (in seconds)
duration = 30

# List to store extracted features for each song
all_features = []

# Selected features
selected_features = [
    'mfcc6_mean', 'mfcc3_mean', 'chroma_stft_var', 'mfcc5_var',
    'mfcc4_var', 'mfcc4_mean', 'mfcc1_mean', 'rms_mean',
    'mfcc19_mean', 'mfcc6_var',
]

# Iterate through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.wav'):  # Assuming the audio files are in WAV format
        file_path = os.path.join(folder_path, filename)

        print(f"Processing file: {filename}")

        # Extract genre from the filename
        genre = filename.split('_')[1].split('.')[0]  # Remove numbers after the dot

        # Load audio file using librosa
        y, sr = librosa.load(file_path, duration=duration)

        print("   Extracting specified features...")

        # Extract specified features
        chroma_stft_var = librosa.feature.chroma_stft(y=y, sr=sr).var()
        rms_mean = librosa.feature.rms(y=y).mean()
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfcc_means = mfcc_features.mean(axis=1)
        mfcc_vars = mfcc_features.var(axis=1)

        print("   Creating a dictionary for features...")

        # Create a dictionary for features
        features = {
            'mfcc6_mean': mfcc_means[5],
            'mfcc3_mean': mfcc_means[2],
            'chroma_stft_var': chroma_stft_var,
            'mfcc5_var': mfcc_vars[4],
            'mfcc4_var': mfcc_vars[3],
            'mfcc4_mean': mfcc_means[3],
            'mfcc1_mean': mfcc_means[0],
            'rms_mean': rms_mean,
            'mfcc19_mean': mfcc_means[18],
            'mfcc6_var': mfcc_vars[5],
            'genre': genre,  # Include the genre in the features
        }


        print("   Appending features to the list...")

        # Append features to the list
        all_features.append(features)

# Create a DataFrame from the list of features
features_df = pd.DataFrame(all_features)

# Save the features to a CSV file
output_path = '/Users/kevin_smith/Desktop/FSU_Relevant_Stuff/fall_2023/CAP5771/project/features_selected.csv'
features_df.to_csv(output_path, index=False)

print(f"Processing complete! Features saved to: {output_path}")
