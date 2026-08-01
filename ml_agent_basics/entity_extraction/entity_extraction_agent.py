"""
Entity Extraction Agent using Gemini API

This module demonstrates an AI agent for Named Entity Recognition (NER) using the Gemini API.
It extracts entities like persons, organizations, locations, and dates from text, then evaluates
performance using precision, recall, and F1-score metrics.

Educational Topics Covered:
- Named Entity Recognition (NER): Identifying and classifying entities in text
- Structured Output: Getting JSON responses from LLMs
- Information Extraction: Pulling specific data from unstructured text
- Evaluation Metrics: Precision, Recall, F1-Score for entity extraction
- Entity Matching: Exact vs fuzzy matching strategies

Classes:
    EntityExtractionAgent: AI agent that extracts named entities using Gemini API
    EntityExtractionEvaluator: Evaluator that calculates Precision, Recall, F1 per entity type

Example Usage:
    >>> agent = EntityExtractionAgent()
    >>> entities = agent.extract_entities("Apple Inc. CEO Tim Cook visited Paris.")
    >>> print(entities)
    [{"text": "Apple Inc.", "type": "ORGANIZATION", ...}, ...]

Requirements:
    - google-generativeai package
    - python-dotenv package
    - GOOGLE_API_KEY or GEMINI_API_KEY environment variable

Author: GDG Workshop - Securing Codebase with ADK and A2A
Version: 1.0
"""

import json
import re
import sys
import os
from typing import List, Dict, Tuple, Set

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


class EntityExtractionAgent:
    """
    An AI agent that performs named entity extraction (NER) using Gemini API.
    
    This agent uses Gemini to identify and classify entities in text as PERSON, ORGANIZATION,
    LOCATION, or DATE. It requests structured JSON output, demonstrating how to get
    formatted responses from LLMs.
    
    Attributes:
        model (genai.GenerativeModel): The Gemini model instance.
    
    Note for Students:
        - NER is a fundamental NLP task used in search engines, chatbots, and analytics
        - Structured output (JSON) makes it easy to integrate with other systems
        - LLMs can do NER without training on labeled data (zero-shot learning)
        - Position tracking (start/end) enables highlighting entities in UI
    """
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the entity extraction agent with a specified Gemini model.
        
        Args:
            model_name (str): Name of the Gemini model to use. Defaults to "gemini-3-flash-preview".
        
        Example:
            >>> agent = EntityExtractionAgent()
            >>> agent_pro = EntityExtractionAgent("gemini-2.5-pro")
        """
        self.model = genai.GenerativeModel(model_name)
        print(f"✓ Initialized EntityExtractionAgent with {model_name}")
    
    def extract_entities(self, text: str) -> List[Dict[str, any]]:
        """
        Extract named entities from text and return them as structured data.
        
        The agent identifies entities of types PERSON, ORGANIZATION, LOCATION, and DATE,
        returning each with its text, type, and position in the original string.
        
        Args:
            text (str): The input text to extract entities from.
            
        Returns:
            List[Dict]: List of entity dictionaries, each containing:
                - 'text' (str): The entity text (e.g., "Apple Inc.")
                - 'type' (str): Entity type (PERSON|ORGANIZATION|LOCATION|DATE)
                - 'start' (int): Starting character position in text
                - 'end' (int): Ending character position in text
        
        Example:
            >>> text = "Tim Cook works at Apple Inc. in California."
            >>> entities = agent.extract_entities(text)
            >>> for e in entities:
            ...     print(f"{e['text']} ({e['type']})")
            Tim Cook (PERSON)
            Apple Inc. (ORGANIZATION)
            California (LOCATION)
        
        Note for Students:
            - Prompt explicitly requests JSON format (structured output)
            - Code handles cases where model wraps JSON in markdown code blocks
            - Position validation ensures entities can be located in original text
            - Returns empty list on error (graceful degradation)
            - In production, consider caching to reduce API calls for repeated texts
        """
        prompt = f"""Extract all named entities from the following text and return them in JSON format.

For each entity, provide:
- "text": the entity text
- "type": one of PERSON, ORGANIZATION, LOCATION, or DATE
- "start": starting character position in the text
- "end": ending character position in the text

Text: "{text}"

Return only a JSON array of entities, no additional text. Example format:
[
  {{"text": "John Smith", "type": "PERSON", "start": 0, "end": 10}},
  {{"text": "Microsoft", "type": "ORGANIZATION", "start": 15, "end": 24}}
]

JSON:"""
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            entities = json.loads(result_text)
            
            # Validate and fix positions if needed
            for entity in entities:
                if 'start' not in entity or 'end' not in entity:
                    # Find positions in text
                    pos = text.find(entity['text'])
                    if pos != -1:
                        entity['start'] = pos
                        entity['end'] = pos + len(entity['text'])
                    else:
                        entity['start'] = 0
                        entity['end'] = 0
            
            return entities
        except Exception as e:
            print(f"Error extracting entities: {e}")
            return []


class EntityExtractionEvaluator:
    """Evaluates entity extraction using Precision, Recall, and F1-score."""
    
    @staticmethod
    def calculate_metrics(true_entities: List[Dict], pred_entities: List[Dict]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1-score for entity extraction.
        
        Args:
            true_entities: Ground truth entities
            pred_entities: Predicted entities
            
        Returns:
            Dictionary with precision, recall, and F1 scores (overall and per entity type)
        """
        # Convert to sets of (text, type, start, end) tuples for comparison
        true_set = set((e['text'].lower(), e['type'], e['start'], e['end']) for e in true_entities)
        pred_set = set((e['text'].lower(), e['type'], e['start'], e['end']) for e in pred_entities)
        
        # Calculate overall metrics
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Calculate per-entity-type metrics
        entity_types = set(e['type'] for e in true_entities + pred_entities)
        per_type_metrics = {}
        
        for etype in entity_types:
            true_type_set = set((e['text'].lower(), e['start'], e['end']) 
                               for e in true_entities if e['type'] == etype)
            pred_type_set = set((e['text'].lower(), e['start'], e['end']) 
                               for e in pred_entities if e['type'] == etype)
            
            tp_type = len(true_type_set & pred_type_set)
            fp_type = len(pred_type_set - true_type_set)
            fn_type = len(true_type_set - pred_type_set)
            
            prec = tp_type / (tp_type + fp_type) if (tp_type + fp_type) > 0 else 0
            rec = tp_type / (tp_type + fn_type) if (tp_type + fn_type) > 0 else 0
            f1_type = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            per_type_metrics[etype] = {
                'precision': round(prec, 4),
                'recall': round(rec, 4),
                'f1': round(f1_type, 4)
            }
        
        return {
            'overall': {
                'precision': round(precision, 4),
                'recall': round(recall, 4),
                'f1': round(f1, 4),
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            },
            'per_type': per_type_metrics
        }


def main():
    """Main function to demonstrate entity extraction and evaluation."""
    
    # Load sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'sample_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Initialize agent and evaluator
    agent = EntityExtractionAgent()
    evaluator = EntityExtractionEvaluator()
    
    print("=" * 80)
    print("ENTITY EXTRACTION AGENT - DEMONSTRATION")
    print("=" * 80)
    
    all_true_entities = []
    all_pred_entities = []
    
    for item in data['texts']:
        print(f"\n{'─' * 80}")
        print(f"Text ID: {item['id']}")
        print(f"{'─' * 80}")
        print(f"\nText: {item['text']}\n")
        
        # Extract entities
        predicted = agent.extract_entities(item['text'])
        true_entities = item['entities']
        
        # Display results
        print("Ground Truth Entities:")
        for entity in true_entities:
            print(f"  - {entity['text']:<30} [{entity['type']}]")
        
        print("\nPredicted Entities:")
        for entity in predicted:
            print(f"  - {entity['text']:<30} [{entity['type']}]")
        
        # Accumulate for overall metrics
        all_true_entities.extend(true_entities)
        all_pred_entities.extend(predicted)
        
        # Calculate metrics for this text
        metrics = evaluator.calculate_metrics(true_entities, predicted)
        
        print(f"\n📊 METRICS FOR THIS TEXT:")
        print(f"  Precision: {metrics['overall']['precision']:.4f}")
        print(f"  Recall:    {metrics['overall']['recall']:.4f}")
        print(f"  F1-Score:  {metrics['overall']['f1']:.4f}")
    
    # Calculate overall metrics
    print(f"\n{'=' * 80}")
    print("OVERALL EVALUATION METRICS:")
    print(f"{'=' * 80}")
    
    overall_metrics = evaluator.calculate_metrics(all_true_entities, all_pred_entities)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Precision:        {overall_metrics['overall']['precision']:.4f}")
    print(f"  Recall:           {overall_metrics['overall']['recall']:.4f}")
    print(f"  F1-Score:         {overall_metrics['overall']['f1']:.4f}")
    print(f"\n  True Positives:   {overall_metrics['overall']['true_positives']}")
    print(f"  False Positives:  {overall_metrics['overall']['false_positives']}")
    print(f"  False Negatives:  {overall_metrics['overall']['false_negatives']}")
    
    print(f"\n📊 PER-ENTITY-TYPE PERFORMANCE:")
    for etype, metrics in overall_metrics['per_type'].items():
        print(f"\n  {etype}:")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
    
    print(f"\n{'=' * 80}\n")


if __name__ == "__main__":
    main()
