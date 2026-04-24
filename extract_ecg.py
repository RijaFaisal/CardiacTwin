import wfdb
import pandas as pd
import os

# These paths assume you are running the script from inside the 'cardiactwin' folder
raw_path = './data/raw/mitbih/' 
output_path = './data/processed/'

# Create the processed folder if it's not there
os.makedirs(output_path, exist_ok=True)

# 100 = Normal, 203 = Arrhythmia, 117 = Bradycardia
records = ['100', '203', '117']

print("Starting extraction...")

for rec in records:
    try:
        # Load the record from your raw folder
        record = wfdb.rdrecord(os.path.join(raw_path, rec))
        
        # Get the first lead (Signal 0)
        signal = record.p_signal[:, 0]
        
        # NORMALIZE: Maps the ECG voltage to 0.0 - 1.0 for Blender
        norm_signal = (signal - signal.min()) / (signal.max() - signal.min())
        
        # Take the first 2000 data points (~5-6 seconds of heartbeats)
        df = pd.DataFrame(norm_signal[:2000])
        
        # Save to your processed folder
        file_name = f'heart_data_{rec}.csv'
        df.to_csv(os.path.join(output_path, file_name), index=False, header=False)
        print(f"✅ Success! Exported: {file_name}")
        
    except Exception as e:
        print(f"❌ Error with record {rec}: {e}")

print("Done! You can now go to Blender.")