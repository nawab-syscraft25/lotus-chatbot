import requests

def search_products(query):
    url = "https://portal.lotuselectronics.com/web-api/home/search_suggestion"
    
    headers = {
        "auth-key": "Web2@!9",
        "end-client": "Lotus-Web",
        "accept": "application/json, text/plain, */*",
        "origin": "https://www.lotuselectronics.com",
        "referer": "https://www.lotuselectronics.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/137.0.0.0 Safari/537.36",
    }

    # Form-data payload
    data = {
        "search_text": query,
        "alias": ""
    }

    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        result = response.json()

        if not result or not isinstance(result, list):
            return []

        return result  # list of products or suggestions

    except Exception as e:
        print(f"[search_products] Error: {e}")
        return []
