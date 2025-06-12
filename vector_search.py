import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from product_utils import get_product_stock_status

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Connect to the existing index
index = pc.Index("lotus-products", host="https://lotus-products-jsy3z1v.svc.aped-4627-b74a.pinecone.io")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_vector_db(query: str, top_k=8):
    vec = embedding_model.encode([query])[0].tolist()
    response = index.query(vector=vec, top_k=top_k, include_metadata=True)
    results = []
    
    for match in response["matches"]:
        metadata = match["metadata"]
        
        # Extract features from the text field
        features = []
        if metadata.get("text"):
            text = metadata["text"]
            # Extract RAM and Storage info
            if "RAM" in text:
                ram_storage = text.split("RAM")[1].split(")")[0].strip()
                features.append(ram_storage)
            # Extract color
            if "Black" in text:
                features.append("Black")
            elif "Gold" in text:
                features.append("Gold")
            # Add 5G if mentioned
            if "5G" in text:
                features.append("5G Connectivity")
        
        # Get product link
        product_link = metadata.get("product_link", "")
        
        # Check stock status and get first image
        is_in_stock = False
        first_image = None
        stock_status = "❌ Out of Stock"
        stock_message = "Product not available online!\nVisit your nearest store to check for best offline deals."
        
        if product_link:
            is_in_stock, first_image, _ = get_product_stock_status(product_link)
            stock_status = "✅ In Stock" if is_in_stock else "❌ Out of Stock"
        
        # Format the product information
        product_info = {
            "name": metadata.get("product_name", ""),
            "price": metadata.get("product_mrp", ""),
            "brand": metadata.get("mpn", ""),
            "link": product_link,
            "features": features,
            "score": match["score"],
            "in_stock": is_in_stock,
            "stock_status": stock_status,
            "stock_message": stock_message if not is_in_stock else "",
            "first_image": first_image or "Image not available"
        }
        
        results.append(product_info)

    # Sort results by:
    # 1. Stock status (in-stock items first)
    # 2. Relevance score (higher scores first)
    sorted_results = sorted(
        results,
        key=lambda x: (
            not x["in_stock"],  # False comes before True, so in-stock items come first
            -x["score"]  # Negative score for descending order
        )
    )
    # print(sorted_results[:5])
    return {
        "type": "general_search",
        "results": sorted_results[:6]  # Return only top 3 results
    }


load_dotenv()

if __name__ == "__main__":
    # Example test query
    test_query = "show me smart phone under 10000"
    print(f"\n[Testing vector search for query: '{test_query}']")
    results = search_vector_db(test_query)
    for result in results["results"]:
        print(f"\nProduct: {result['name']}")
        print(f"Product Link: {result['link']}")
        print(f"Price: ₹{result['price']}")
        print(f"Stock Status: {result['stock_status']}")
        if not result['in_stock']:
            print(f"Note: {result['stock_message']}")
        print(f"Product Image: {result['first_image']}")
        print("Features:", result['features'])
        print("-" * 50)