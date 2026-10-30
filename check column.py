import pandas as pd
df = pd.read_excel('scholarship dataset 72.xlsx', sheet_name='Scholarships_Actual')
print("--- ALL DATASET COLUMNS ---")
for col in df.columns:
    print(f"- {col}")