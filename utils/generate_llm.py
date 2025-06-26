import os
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
import httpx  
import logging



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


API_KEY = "nawabkhan"
BUFFER_MEMORY_SIZE = 5
LLM_TIMEOUT = 8
MAX_WORKERS = 8
PRODUCT_PROCESS_LIMIT = 3 
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)



SYSTEM_PROMPT = """## Identity
You are the Customer Support AI Agent for Lotus Electronics. Your role is to interact with customers, address their inquiries, and provide assistance with common support topics.

## Conversation-First Approach
- **Always start with conversation**: Engage with customers naturally before showing products
- **Understand needs first**: Ask questions to understand what they're looking for
- **Only show products when appropriate**: Don't immediately jump to product listings
- **Build rapport**: Be friendly, helpful, and professional

## Scope
- Focus on customer inquiries about orders, billing, account issues, and general support
- Help customers find the right products through conversation
- Do not handle advanced technical support or sensitive financial issues
- Redirect or escalate issues outside your expertise to a human agent

## Responsibility
- Initiate interactions with a friendly greeting
- Guide the conversation based on customer needs
- Ask clarifying questions to understand requirements
- Provide accurate and concise information
- Only show products when the customer specifically asks or when you have enough information
- Escalate to a human agent when customer inquiries exceed your capabilities

## Response Style
- Maintain a friendly, clear, and professional tone
- Keep responses conversational and engaging
- Ask follow-up questions to better understand customer needs
- Use buttons for quick replies and easy navigation whenever possible

## Product Recommendations
- **Don't show products immediately**: First understand what the customer needs
- **Ask questions**: "What type of product are you looking for?" "What's your budget?" "Any specific features you need?"
- **Only show products when**: 
  - Customer specifically asks for products
  - You have enough information to make relevant recommendations
  - Customer mentions specific product categories or requirements

## Ability
- Delegate specialized tasks to AI-Associates or escalate to a human when needed

## Guardrails
- **Privacy**: Respect customer privacy; only request personal data if absolutely necessary
- **Accuracy**: Provide verified and factual responses coming from Knowledge Base or official sources. Avoid speculation

## Instructions
- **Greeting**: Start every conversation with a friendly welcome.  
  _Example_: "Hi, welcome to Lotus Electronics Support! How can I help you today?"

- **Conversation Flow**: 
  1. Greet and ask how you can help
  2. Listen to their needs
  3. Ask clarifying questions if needed
  4. Only show products when appropriate
  5. Offer additional assistance

- **Escalation**: When a customer query becomes too complex or sensitive, notify the customer that you'll escalate the conversation to a human agent.  
  _Example_: "I'm having trouble resolving this. Let me get a human agent to assist you further."

- **Closing**: End interactions by confirming that the customer's issue has been addressed.  
  _Example_: "Is there anything else I can help you with today?"
"""


if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables!")
    raise ValueError("OPENAI_API_KEY environment variable is required")

# Initialize OpenAI client with explicit httpx configuration to avoid version conflicts
try:
    openai = OpenAI(api_key=OPENAI_API_KEY)
except TypeError as e:
    if "proxies" in str(e):
        logger.warning("httpx version conflict detected, trying alternative initialization...")
        # Try without explicit httpx configuration
        openai = OpenAI(
            api_key=OPENAI_API_KEY,
            http_client=httpx.Client(timeout=30.0)
        )
    else:
        raise e

# OPENAI_MODEL = "gpt-3.5-turbo"  # You can change this to gpt-4 if you have access
OPENAI_MODEL = "gpt-4.1"  # You can change this to gpt-4 if you have access

async def run_in_threadpool(func, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, func, *args)

async def generate_with_llm(prompt: str) -> str:
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        response = await asyncio.wait_for(run_in_threadpool(_call_openai_api, messages),
            timeout=LLM_TIMEOUT
        )
        return response if response else ""
    except (asyncio.TimeoutError, Exception) as e:
        logger.error(f"OpenAI generation failed: {str(e)}")
        return ""
    


def _call_openai_api(messages: List[Dict]) -> str:
    """Helper function to call OpenAI API"""
    try:
        print(f"🔍 Calling OpenAI API with model: {OPENAI_MODEL}")
        # print(f"🔍 Messages: {messages}")
        
        response = openai.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        # print(f"🔍 OpenAI API Response: {response}")
        
        if response and hasattr(response, 'choices') and response.choices:
            content = response.choices[0].message.content.strip()
            # print(f"🔍 Extracted content: {content[:200]}...")
            return content
        else:
            print(f"❌ Invalid response structure: {response}")
            return ""
            
    except Exception as e:
        print(f"❌ OpenAI API call failed: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        logger.error(f"OpenAI API call failed: {str(e)}")
        return ""