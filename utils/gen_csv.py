import os
from pathlib import Path
import pandas as pd

file_dir = "/Users/adees/Code/neural_granular_synthesis/datasets/OceanWaves/samples/1secs/full"
csv_path = "./data_csv.csv"

file_list = []
for root, dirs, files in os.walk(file_dir):
    for file in files:
        if file.endswith('.flac') or file.endswith('.wav') or file.endswith('.mp3') in root:
            file_list.append(os.path.join(root, file))
print(f"file length:{len(file_list)}")

csv_path = Path(csv_path)
if not csv_path.parent.exists():
    csv_path.parent.mkdir(parents=True)
    
data = pd.DataFrame(file_list)
data.to_csv(csv_path, index=False, header=False)