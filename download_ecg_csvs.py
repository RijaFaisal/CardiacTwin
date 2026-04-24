import wfdb
import pandas as pd

# The record name (e.g., if files are 'patient1.dat', 'patient1.hea', 'patient1.atr')
record_name = 'patient1' 

# Read the annotation file
# The 'extension' argument must match the file extension (e.g., 'atr', 'qrs', or '123')
annotation = wfdb.rdann(record_name, extension='atr')

# Extract the key arrays
stamps = annotation.sample   # The exact index/sample number of the event
symbols = annotation.symbol  # The character code indicating what the event is

# Combine them into a Pandas DataFrame for easy viewing and filtering
df_annotations = pd.DataFrame({
    'Sample_Index': stamps,
    'Symbol': symbols
})






print(df_annotations.head(15))