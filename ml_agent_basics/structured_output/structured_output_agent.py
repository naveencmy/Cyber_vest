"""
Structured Output Agent using Gemini Function Calling API

This module demonstrates how to use Gemini's function calling feature to get reliable,
structured outputs from AI agents. It showcases practical use cases like extracting
product information, parsing contact details, and formatting data.

Educational Topics Covered:
- Function Calling: Teaching LLMs to output structured data via function schemas
- Structured Output: Ensuring consistent, parseable responses
- Schema Definition: Defining data structures with JSON Schema
- Input Validation: Ensuring AI outputs match expected formats
- Practical Applications: Product info extraction, contact parsing, event scheduling

Classes:
    StructuredOutputAgent: AI agent that uses function calling for structured responses
    DataFormatter: Utilities for formatting and validating structured data

Example Usage:
    >>> agent = StructuredOutputAgent()
    >>> product = agent.extract_product_info("iPhone 15 Pro costs $999 with 256GB storage")
    >>> print(product)
    {"name": "iPhone 15 Pro", "price": 999.0, "storage": "256GB"}

Requirements:
    - google-generativeai package
    - python-dotenv package
    - GOOGLE_API_KEY or GEMINI_API_KEY environment variable

Author: GDG Workshop - Securing Codebase with ADK and A2A
Version: 1.0
"""

import json
import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("No Gemini API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file.")

genai.configure(api_key=GEMINI_API_KEY)


class StructuredOutputAgent:
    """
    An AI agent that uses Gemini's function calling to extract structured data.
    
    This agent demonstrates how to use function calling as a reliable way to get
    structured, parseable outputs from LLMs. Instead of parsing free-form text,
    we define function schemas that the model must follow.
    
    Attributes:
        model (genai.GenerativeModel): The Gemini model with function calling enabled.
    
    Note for Students:
        - Function calling ensures outputs match your defined schema
        - More reliable than parsing free-form text responses
        - Widely used in production AI systems (chatbots, data extraction, APIs)
        - Schema = contract between your code and the AI
    """
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the structured output agent.
        
        Args:
            model_name (str): Name of the Gemini model to use.
        
        Example:
            >>> agent = StructuredOutputAgent()
        """
        # Define function schemas for structured output
        self.tools = [
            genai.protos.Tool(
                function_declarations=[
                    genai.protos.FunctionDeclaration(
                        name="extract_product_info",
                        description="Extract product information from text including name, price, and features",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Product name"),
                                "price": genai.protos.Schema(type=genai.protos.Type.NUMBER, description="Product price in USD"),
                                "currency": genai.protos.Schema(type=genai.protos.Type.STRING, description="Currency code (e.g., USD, EUR)"),
                                "storage": genai.protos.Schema(type=genai.protos.Type.STRING, description="Storage capacity if applicable"),
                                "color": genai.protos.Schema(type=genai.protos.Type.STRING, description="Product color if mentioned"),
                                "features": genai.protos.Schema(
                                    type=genai.protos.Type.ARRAY,
                                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                                    description="List of key features"
                                )
                            },
                            required=["name", "price", "currency"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="extract_contact_info",
                        description="Extract contact information from text including name, email, phone, and address",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "name": genai.protos.Schema(type=genai.protos.Type.STRING, description="Person's full name"),
                                "email": genai.protos.Schema(type=genai.protos.Type.STRING, description="Email address"),
                                "phone": genai.protos.Schema(type=genai.protos.Type.STRING, description="Phone number"),
                                "company": genai.protos.Schema(type=genai.protos.Type.STRING, description="Company name"),
                                "title": genai.protos.Schema(type=genai.protos.Type.STRING, description="Job title"),
                                "address": genai.protos.Schema(type=genai.protos.Type.STRING, description="Physical address")
                            },
                            required=["name"]
                        )
                    ),
                    genai.protos.FunctionDeclaration(
                        name="extract_event_info",
                        description="Extract event information including title, date, time, location, and attendees",
                        parameters=genai.protos.Schema(
                            type=genai.protos.Type.OBJECT,
                            properties={
                                "title": genai.protos.Schema(type=genai.protos.Type.STRING, description="Event title"),
                                "date": genai.protos.Schema(type=genai.protos.Type.STRING, description="Event date (YYYY-MM-DD format)"),
                                "time": genai.protos.Schema(type=genai.protos.Type.STRING, description="Event time"),
                                "location": genai.protos.Schema(type=genai.protos.Type.STRING, description="Event location"),
                                "attendees": genai.protos.Schema(
                                    type=genai.protos.Type.ARRAY,
                                    items=genai.protos.Schema(type=genai.protos.Type.STRING),
                                    description="List of attendees"
                                ),
                                "description": genai.protos.Schema(type=genai.protos.Type.STRING, description="Event description")
                            },
                            required=["title", "date"]
                        )
                    )
                ]
            )
        ]
        
        self.model = genai.GenerativeModel(model_name, tools=self.tools)
        print(f"✓ Initialized StructuredOutputAgent with {model_name}")
        print(f"✓ Loaded {len(self.tools[0].function_declarations)} function schemas")
    
    def extract_product_info(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract product information from text using function calling.
        
        Args:
            text (str): Text containing product information.
        
        Returns:
            Dict: Structured product information or None if extraction fails.
        
        Example:
            >>> info = agent.extract_product_info("MacBook Pro 16-inch is $2499 with M3 chip")
            >>> print(info['name'], info['price'])
            MacBook Pro 16-inch 2499.0
        """
        prompt = f"Extract product information from the following text:\n\n{text}"
        return self._call_function(prompt, "extract_product_info")
    
    def extract_contact_info(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract contact information from text using function calling.
        
        Args:
            text (str): Text containing contact information.
        
        Returns:
            Dict: Structured contact information or None if extraction fails.
        
        Example:
            >>> info = agent.extract_contact_info("John Doe, john@example.com, (555) 123-4567")
            >>> print(info['email'])
            john@example.com
        """
        prompt = f"Extract contact information from the following text:\n\n{text}"
        return self._call_function(prompt, "extract_contact_info")
    
    def extract_event_info(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract event information from text using function calling.
        
        Args:
            text (str): Text containing event information.
        
        Returns:
            Dict: Structured event information or None if extraction fails.
        
        Example:
            >>> info = agent.extract_event_info("Team meeting on 2024-03-15 at 2 PM in Conference Room A")
            >>> print(info['title'], info['date'])
            Team meeting 2024-03-15
        """
        prompt = f"Extract event information from the following text:\n\n{text}"
        return self._call_function(prompt, "extract_event_info")
    
    def _call_function(self, prompt: str, expected_function: str) -> Optional[Dict[str, Any]]:
        """
        Internal method to call the model and extract function call results.
        
        Args:
            prompt (str): The prompt to send to the model.
            expected_function (str): Name of the expected function to be called.
        
        Returns:
            Dict: The function call arguments as a dictionary, or None on failure.
        
        Note for Students:
            - Model response contains function_call object with name and args
            - Args are already parsed as a dictionary (no JSON parsing needed!)
            - This is much more reliable than parsing free-form text
        """
        try:
            response = self.model.generate_content(prompt)
            
            # Check if model made a function call
            if response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                if part.function_call:
                    function_call = part.function_call
                    if function_call.name == expected_function:
                        # Convert args to dict
                        return dict(function_call.args)
            
            print(f"Warning: No {expected_function} function call in response")
            return None
            
        except Exception as e:
            print(f"Error calling function: {e}")
            return None


class DataFormatter:
    """
    Utility class for formatting and validating structured data.
    
    Provides helper methods to display extracted data in human-readable formats.
    """
    
    @staticmethod
    def format_product(product: Dict[str, Any]) -> str:
        """Format product information for display."""
        output = [f"📦 Product: {product.get('name', 'Unknown')}"]
        output.append(f"💰 Price: {product.get('currency', '$')}{product.get('price', 0):.2f}")
        
        if 'storage' in product:
            output.append(f"💾 Storage: {product['storage']}")
        if 'color' in product:
            output.append(f"🎨 Color: {product['color']}")
        if 'features' in product and product['features']:
            output.append(f"✨ Features: {', '.join(product['features'])}")
        
        return "\n".join(output)
    
    @staticmethod
    def format_contact(contact: Dict[str, Any]) -> str:
        """Format contact information for display."""
        output = [f"👤 Name: {contact.get('name', 'Unknown')}"]
        
        if 'email' in contact:
            output.append(f"📧 Email: {contact['email']}")
        if 'phone' in contact:
            output.append(f"📱 Phone: {contact['phone']}")
        if 'company' in contact:
            output.append(f"🏢 Company: {contact['company']}")
        if 'title' in contact:
            output.append(f"💼 Title: {contact['title']}")
        if 'address' in contact:
            output.append(f"📍 Address: {contact['address']}")
        
        return "\n".join(output)
    
    @staticmethod
    def format_event(event: Dict[str, Any]) -> str:
        """Format event information for display."""
        output = [f"📅 Event: {event.get('title', 'Untitled')}"]
        output.append(f"📆 Date: {event.get('date', 'TBD')}")
        
        if 'time' in event:
            output.append(f"🕐 Time: {event['time']}")
        if 'location' in event:
            output.append(f"📍 Location: {event['location']}")
        if 'attendees' in event and event['attendees']:
            output.append(f"👥 Attendees: {', '.join(event['attendees'])}")
        if 'description' in event:
            output.append(f"📝 Description: {event['description']}")
        
        return "\n".join(output)


def main():
    """Main function to demonstrate structured output extraction."""
    
    # Load sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'sample_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
   # Initialize agent and formatter
    agent = StructuredOutputAgent()
    formatter = DataFormatter()
    
    print("\n" + "=" * 80)
    print("STRUCTURED OUTPUT AGENT - DEMONSTRATION")
    print("=" * 80)
    
    # Product extraction examples
    print("\n" + "─" * 80)
    print("📦 PRODUCT INFORMATION EXTRACTION")
    print("─" * 80)
    
    for item in data['products']:
        print(f"\nInput Text: \"{item['text']}\"")
        product = agent.extract_product_info(item['text'])
        if product:
            print(f"\n{formatter.format_product(product)}")
        print()
    
    # Contact extraction examples
    print("─" * 80)
    print("👤 CONTACT INFORMATION EXTRACTION")
    print("─" * 80)
    
    for item in data['contacts']:
        print(f"\nInput Text: \"{item['text']}\"")
        contact = agent.extract_contact_info(item['text'])
        if contact:
            print(f"\n{formatter.format_contact(contact)}")
        print()
    
    # Event extraction examples
    print("─" * 80)
    print("📅 EVENT INFORMATION EXTRACTION")
    print("─" * 80)
    
    for item in data['events']:
        print(f"\nInput Text: \"{item['text']}\"")
        event = agent.extract_event_info(item['text'])
        if event:
            print(f"\n{formatter.format_event(event)}")
        print()
    
    print("=" * 80)
    print("\n✨ Key Takeaways:")
    print("  • Function calling provides reliable, structured output")
    print("  • Schema validation ensures data consistency")
    print("  • No need to parse free-form text responses")
    print("  • Perfect for integrating AI into production systems\n")


if __name__ == "__main__":
    main()
