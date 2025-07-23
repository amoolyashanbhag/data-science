import pandas as pd
import requests
import matplotlib.pyplot as plt

# URL of the World Bank data
url = "https://api.worldbank.org/v2/en/indicator/SP.POP.TOTL?downloadformat=csv"

# Fetch the data
response = requests.get(url)
with open("world_bank_data.zip", "wb") as f:
    f.write(response.content)

# Unzip the file
import zipfile
import os

with zipfile.ZipFile("world_bank_data.zip", "r") as zip_ref:
    zip_ref.extractall("world_bank_data")

# Locate and load the CSV file
csv_files = [file for file in os.listdir("world_bank_data") if file.endswith(".csv")]
data_file = f"world_bank_data/{csv_files[0]}"

# Read the data
data = pd.read_csv(data_file, skiprows=4)

# Filter for the latest data (e.g., 2023)
latest_data = data[["Country Name", "2023"]].dropna()

# Create a bar chart for population distribution
plt.figure(figsize=(10, 6))
plt.bar(latest_data["Country Name"].head(10), latest_data["2023"].head(10))
plt.xlabel("Country")
plt.ylabel("Population")
plt.title("Top 10 Countries by Population in 2023")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
