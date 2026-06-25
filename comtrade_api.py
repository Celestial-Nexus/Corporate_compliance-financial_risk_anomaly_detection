import requests
import pandas as pd

url = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"

params = {
    'reporterCode': 842, # 842 is the numeric UN M49 code for the USA
    'partnerCode': 156,  # 156 is the code for China
    'period': 2023,      # The year of trade
    'flowCode': 'M',     # 'X' for Exports, 'M' for Imports
    'cmdCode': 'AG2',    # 'AG2' gets all 2-digit HS product chapters
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()
    df = pd.DataFrame(data['data'])

    top_exports = df[['cmdCode', 'primaryValue']].sort_values('primaryValue', ascending=False)
    
    print("Top US Exports to China in 2023 (by HS Code):")
    print(top_exports.head())

else:
    print(f"Failed to fetch data: {response.status_code}")