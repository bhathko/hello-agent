import os
import requests


def get_weather(city: str) -> str:
    """
    Get weather information using Google Maps Platform Weather API.
    
    This tool performs two steps:
    1. Geocoding: Converts the city name into Latitude/Longitude.
    2. Weather Lookup: Uses coordinates to get current conditions.
    
    Note: The Google Weather API has regional limitations (e.g., Beijing is not currently supported).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY is not configured."

    # 1. Geocoding: Convert city name to Lat/Lng
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={city}&key={api_key}"
    
    try:
        geo_response = requests.get(geocode_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if geo_data["status"] != "OK":
            status = geo_data.get("status", "UNKNOWN")
            return f"Error: Could not find location information for city '{city}' (Status: {status})."

        location = geo_data["results"][0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]

        # 2. Weather Lookup: Fetch current conditions
        weather_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
        params = {
            "key": api_key,
            "location.latitude": lat,
            "location.longitude": lng,
            "languageCode": "en",
            "unitsSystem": "METRIC"
        }

        weather_response = requests.get(weather_url, params=params)
        
        # Handle regional support limitations (Google returns 404 for unsupported regions)
        if weather_response.status_code == 404:
            return f"Sorry, the Google Weather API does not currently support the region for '{city}'. Please try searching for other international cities (e.g., Tokyo, New York, London, etc.)."
            
        weather_response.raise_for_status()
        weather_data = weather_response.json()

        # 3. Extract and format weather data
        weather_condition_obj = weather_data.get("weatherCondition", {})
        condition = weather_condition_obj.get("description", {}).get("text", "Unknown")
        
        temp_obj = weather_data.get("temperature", {})
        temperature = temp_obj.get("degrees", "Unknown")
        
        humidity = weather_data.get("relativeHumidity", "Unknown")

        return f"Current weather in {city}: {condition}, Temperature: {temperature}°C, Relative Humidity: {humidity}%"

    except requests.exceptions.RequestException as e:
        return f"Error: Network issue occurred while calling Google API - {e}"
    except (KeyError, IndexError) as e:
        return f"Error: Failed to parse Google API response data - {e}"
