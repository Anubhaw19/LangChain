import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from datetime import datetime, timedelta

def get_weather_data(city):
    city_url = city.lower().replace(" ", "-")

    current_url = f"https://www.timeanddate.com/weather/india/{city_url}"
    hourly_url = f"https://www.timeanddate.com/weather/india/{city_url}/hourly"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    today = datetime.now().date()

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
            "date": today.strftime('%Y-%m-%d'),
            "time": "Now",
            "temperature": temperature.replace('\xa0', ' '),
            "description": description,
            "precip_chance": "N/A",
            "precip_amount": "N/A"
        }

    except Exception as e:
        current_weather = {
            "city": city.title(),
            "date": today.strftime('%Y-%m-%d'),
            "time": "Now",
            "temperature": "N/A",
            "description": f"Error: {str(e)}",
            "precip_chance": "N/A",
            "precip_amount": "N/A"
        }

    # --- Hourly Forecast (Next 12 Hours) ---
    forecast_data = []
    try:
        r = requests.get(hourly_url, headers=headers)
        soup = BeautifulSoup(r.text, "html.parser")

        table = soup.find("table", class_="zebra tb-wt fw va-m tb-hover")
        rows = table.find_all("tr")[1:13]  # Next 12 hourly entries

        forecast_date = today
        previous_hour = None

        for row in rows:
            time_cell = row.find("th")
            raw_time = time_cell.get_text(strip=True) if time_cell else "N/A"

            # Extract HH.MM
            match = re.match(r'^(\d{1,2})\.(\d{2})', raw_time)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2))
                time_str = f"{hour:02d}.{minute:02d}"

                # Check for date rollover
                if previous_hour is not None and hour < previous_hour:
                    forecast_date += timedelta(days=1)
                previous_hour = hour

                cells = row.find_all("td")
                if len(cells) >= 9:
                    temperature = cells[1].get_text(strip=True).replace('\xa0', ' ')
                    description = cells[2].get_text(strip=True)
                    precip_chance = cells[7].get_text(strip=True)
                    # precip_amount = cells[8].get_text(strip=True)
                    precip_amount_raw = cells[8].get_text(strip=True)
                    match_rain = re.search(r'([\d.]+)\s*mm\s*\(rain\)', precip_amount_raw)
                    precip_amount = match_rain.group(1) + " mm" if match_rain else "0 mm"


                    forecast_data.append({
                        "city": city.title(),
                        "date": forecast_date.strftime('%Y-%m-%d'),
                        "time": time_str,
                        "temperature": temperature,
                        "description": description,
                        "precip_chance": precip_chance,
                        "precip_amount": precip_amount
                    })
    except Exception as e:
        forecast_data = [{
            "city": city.title(),
            "date": today.strftime('%Y-%m-%d'),
            "time": "N/A",
            "temperature": "N/A",
            "description": f"Error: {str(e)}",
            "precip_chance": "N/A",
            "precip_amount": "N/A"
        }]

    return pd.DataFrame([current_weather] + forecast_data)


# --- Cities to Scrape ---
cities = ["mumbai", "bangalore", "pune", "delhi", "amritsar", "ludhiana"]

# Collect weather data for all cities
all_weather_data = pd.concat([get_weather_data(city) for city in cities], ignore_index=True)

# Save to CSV
filename = "weather_6_cities_with_precip.csv"
all_weather_data.to_csv(filename, index=False)
print(f"\n Weather data saved to: {filename}")

# Preview
print("\n Preview:")
print(all_weather_data.head(15))
