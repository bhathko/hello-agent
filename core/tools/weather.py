import os
import requests


def get_weather(city: str) -> str:
    """
    使用 Google Maps Platform 天氣 API 取得天氣資訊。
    先進行地理編碼，再取得當前天氣狀況。
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return "錯誤：GOOGLE_API_KEY 未設定。"

    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={city}&key={api_key}"

    try:
        geo_response = requests.get(geocode_url)
        geo_response.raise_for_status()
        geo_data = geo_response.json()

        if geo_data["status"] != "OK":
            status = geo_data.get("status", "UNKNOWN")
            return f"錯誤：找不到城市 '{city}' 的位置資訊（狀態碼：{status}）。"

        location = geo_data["results"][0]["geometry"]["location"]
        lat, lng = location["lat"], location["lng"]

        weather_url = "https://weather.googleapis.com/v1/currentConditions:lookup"
        params = {
            "key": api_key,
            "location.latitude": lat,
            "location.longitude": lng,
            "languageCode": "zh-TW",
            "unitsSystem": "METRIC"
        }

        weather_response = requests.get(weather_url, params=params)

        if weather_response.status_code == 404:
            return f"抱歉，Google 天氣 API 目前不支援 '{city}' 所在的地區。請嘗試其他國際城市。"

        weather_response.raise_for_status()
        weather_data = weather_response.json()

        condition = weather_data.get("weatherCondition", {}).get("description", {}).get("text", "未知")
        temperature = weather_data.get("temperature", {}).get("degrees", "未知")
        humidity = weather_data.get("relativeHumidity", "未知")

        return f"{city} 目前天氣：{condition}，氣溫：{temperature}°C，相對濕度：{humidity}%"

    except requests.exceptions.RequestException as e:
        return f"錯誤：呼叫 Google API 時發生網路問題 - {e}"
    except (KeyError, IndexError) as e:
        return f"錯誤：解析 Google API 回應資料失敗 - {e}"
