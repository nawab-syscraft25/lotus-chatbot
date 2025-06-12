import requests
import json
from typing import Optional, List, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductImageExtractor:
    def __init__(self):
        self.api_url = "https://portal.lotuselectronics.com/web-api/home/product_detail"
        self.headers = {
            'auth-key': 'Web2@!9',
            'end-client': 'Lotus-Web',
            'origin': 'https://www.lotuselectronics.com',
            'referer': 'https://www.lotuselectronics.com/',
            'accept': 'application/json, text/plain, */*'
        }

    def extract_product_id_from_url(self, url: str) -> Optional[str]:
        """Extract product ID from the product URL."""
        try:
            # Split URL by '/' and get the last part
            parts = url.strip('/').split('/')
            return parts[-1]
        except Exception as e:
            logger.error(f"Error extracting product ID from URL: {e}")
            return None

    def extract_category_from_url(self, url: str) -> Optional[str]:
        """Extract category from the product URL."""
        try:
            # Split URL by '/' and get the second last part
            parts = url.strip('/').split('/')
            return parts[-2]
        except Exception as e:
            logger.error(f"Error extracting category from URL: {e}")
            return None

    def get_product_images(self, product_url: str) -> List[str]:
        """
        Get product images from the API.
        
        Args:
            product_url (str): The product URL
            
        Returns:
            List[str]: List of image URLs
        """
        try:
            # Extract product ID and category from URL
            product_id = self.extract_product_id_from_url(product_url)
            category = self.extract_category_from_url(product_url)
            
            if not product_id or not category:
                logger.error("Could not extract product ID or category from URL")
                return []

            # Prepare form data
            form_data = {
                'product_id': product_id,
                'cat_name': f"/product/{category}",
                'product_name': product_url.split('/')[-1].lower()
            }

            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=form_data
            )

            # Check if request was successful (200 or 201)
            if response.status_code in [200, 201]:
                data = response.json()
                logger.info(f"API Response: {json.dumps(data, indent=2)}")
                
                # Extract image URLs from response
                if (data.get('data') and 
                    data['data'].get('product_detail') and 
                    data['data']['product_detail'].get('product_image')):
                    
                    # Clean image URLs (remove escape characters)
                    image_urls = [
                        url.replace('\\', '') 
                        for url in data['data']['product_detail']['product_image']
                    ]
                    return image_urls
                
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return []

        except requests.RequestException as e:
            logger.error(f"Error making API request: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []

    def get_product_details(self, product_url: str) -> Dict:
        """
        Get complete product details including images.
        
        Args:
            product_url (str): The product URL
            
        Returns:
            Dict: Product details including images
        """
        try:
            # Extract product ID and category from URL
            product_id = self.extract_product_id_from_url(product_url)
            category = self.extract_category_from_url(product_url)
            
            if not product_id or not category:
                logger.error("Could not extract product ID or category from URL")
                return {}

            # Prepare form data
            form_data = {
                'product_id': product_id,
                'cat_name': f"/product/{category}",
                'product_name': product_url.split('/')[-1].lower()
            }

            # Make API request
            response = requests.post(
                self.api_url,
                headers=self.headers,
                data=form_data
            )

            # Check if request was successful (200 or 201)
            if response.status_code in [200, 201]:
                data = response.json()
                logger.info(f"API Response: {json.dumps(data, indent=2)}")
                
                if data.get('data') and data['data'].get('product_detail'):
                    product_detail = data['data']['product_detail']
                    
                    # Clean image URLs
                    if product_detail.get('product_image'):
                        product_detail['product_image'] = [
                            url.replace('\\', '') 
                            for url in product_detail['product_image']
                        ]
                    
                    return product_detail
                
            logger.error(f"API request failed with status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return {}

        except requests.RequestException as e:
            logger.error(f"Error making API request: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing API response: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = ProductImageExtractor()
    
    # Example product URL
    product_url = "https://www.lotuselectronics.com/product/hd-led-tv/samsung-hd-led-tv-80-cm-32-inches-ua32t4310-black/32493"
    
    # Get product images
    images = extractor.get_product_images(product_url)
    print("Product Images:")
    for img in images:
        print(f"- {img}")
    
    # Get complete product details
    product_details = extractor.get_product_details(product_url)
    print("\nProduct Details:")
    print(json.dumps(product_details, indent=2)) 