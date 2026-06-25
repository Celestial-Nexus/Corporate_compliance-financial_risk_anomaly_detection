import requests
import matplotlib.pyplot as plt

def fetch_world_bank_population(country_code="IN"):
    
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/SP.POP.TOTL"
    
    params = {
        "format": "json",
        "per_page": 15
    }
    
    print(f"Fetching data from: {url}")
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Failed to retrieve data. HTTP Status: {response.status_code}")
        return None, None

    data = response.json()
    
    if len(data) < 2:
        print("No data payload found in response.")
        return None, None
        
    records = data[1]
    
    years = []
    populations = []
    
    for record in records:
        if record.get('value') is not None:
            years.append(int(record['date']))
            populations.append(record['value'])
            
    years.reverse()
    populations.reverse()
    
    return years, populations

def main():
    country = "IN"
    years, populations = fetch_world_bank_population(country_code=country)
    
    if years and populations:
        print(f"\nSuccessfully retrieved {len(years)} records. Launching visualizer...\n")
        
        # Instantiate the visualizer to plot the time-series data
        plt.figure(figsize=(10, 6))
        plt.plot(years, populations, marker='o', linestyle='-', color='indigo', linewidth=2)
        plt.title(f"Total Population Over Time (Country Code: {country})", fontsize=14)
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Total Population", fontsize=12)
        
        plt.ticklabel_format(style='plain', axis='y')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.show()

if __name__ == "__main__":
    main()