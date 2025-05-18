import pandas as pd

# Load dataset
file_path = "Cotton_2020.csv"
df = pd.read_csv(file_path)

# Convert date to datetime format
df['arrival_date'] = pd.to_datetime(df['arrival_date'], format='%d/%m/%Y')

# Select relevant columns
df = df[['state', 'district', 'market', 'arrival_date', 'modal_price']]

# Save processed dataset
df.to_csv('processed_cotton_prices.csv', index=False)
print("Dataset 'processed_cotton_prices.csv' created successfully!")
