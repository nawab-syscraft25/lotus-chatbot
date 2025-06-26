import re



def extract_price_filter(query: str):
    query = query.replace("₹", "").replace(",", "").lower()

    if match := re.search(r"(under|below)\s*(\d+)", query):
        return {"$lte": int(match.group(2))}
    elif match := re.search(r"(above|over)\s*(\d+)", query):
        return {"$gte": int(match.group(2))}
    elif match := re.search(r"between\s*(\d+)\s*and\s*(\d+)", query):
        return {"$gte": int(match.group(1)), "$lte": int(match.group(2))}
    return None