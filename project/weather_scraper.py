import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def get_weather_data(city):
    city_url = city.lower().replace(" ", "-")

    current_url = f"https://www.timeanddate.com/weather/india/{city_url}"
    hourly_url = f"https://www.timeanddate.com/weather/india/{city_url}/hourly"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    # --- Current Weather ---
    current_weather = {}
    try:
        r = requests.get(current_url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")

        temp_div = soup.find("div", class_="h2")
        temperature = temp_div.get_text(strip=True) if temp_div else "N/A"

        desc_p = soup.select_one("div.bk-focus__qlook p")
        description = desc_p.get_text(strip=True) if desc_p else "N/A"

        current_weather = {
            "city": city.title(),
            "time": "Now",
            "temperature": temperature.replace('\xa0', ' '),
            "description": description
        }

    except Exception as e:
        current_weather = {
            "city": city.title(),
            "time": "Now",
            "temperature": "N/A",
            "description": f"Error: {str(e)}"
        }

    # --- Hourly Forecast (Next 12 Hours) ---
    forecast_data = []
    try:
        r = requests.get(hourly_url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")

        table = soup.find("table", class_="zebra tb-wt fw va-m tb-hover")
        rows = table.find_all("tr")[1:13]

        for row in rows:
            # time_cell = row.find("th")
            # time = time_cell.get_text(strip=True) if time_cell else "N/A"

            time_cell = row.find("th")
            raw_time = time_cell.get_text(strip=True) if time_cell else "N/A"
            match = re.match(r'^(\d{1,2}\.\d{2})', raw_time)
            time = match.group(1) if match else raw_time

            cells = row.find_all("td")
            if len(cells) >= 3:
                temperature = cells[1].get_text(strip=True).replace('\xa0', ' ')
                description = cells[2].get_text(strip=True)

                forecast_data.append({
                    "city": city.title(),
                    "time": time,
                    "temperature": temperature,
                    "description": description
                })
    except Exception as e:
        forecast_data = [{
            "city": city.title(),
            "time": "N/A",
            "temperature": "N/A",
            "description": f"Error: {str(e)}"
        }]

    # Return one combined DataFrame
    return pd.DataFrame([current_weather] + forecast_data)


# List of cities
cities = ["mumbai", "bangalore", "pune", "delhi", "amritsar", "ludhiana"]

# Combine data for all cities
all_weather_data = pd.concat([get_weather_data(city) for city in cities], ignore_index=True)

# Save to CSV
filename = "weather.csv"
all_weather_data.to_csv(filename, index=False)


# Optional: Print preview
print("\n Preview:")
print(all_weather_data.head(15))
