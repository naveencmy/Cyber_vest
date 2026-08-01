"""
Text Classification Agent using Gemini API

This module demonstrates an AI agent for sentiment classification using few-shot learning.
It showcases how to use the Gemini API for classification tasks and evaluate performance
using standard metrics like accuracy and F1-score.

Educational Topics Covered:
- Few-Shot Learning: Teaching AI by providing examples in the prompt
- Sentiment Analysis: Classifying text as positive, negative, or neutral
- Classification Metrics: Accuracy, F1-Score (Macro vs Micro), Confusion Matrix
- Prompt Engineering: Crafting examples to guide model behavior
- Evaluation Best Practices: Using multiple metrics for comprehensive assessment

Classes:
    ClassificationAgent: AI agent that classifies text using Gemini API with few-shot learning
    ClassificationEvaluator: Evaluator that calculates Accuracy, F1-Score, and Confusion Matrix

Example Usage:
    >>> agent = ClassificationAgent()
    >>> agent.train(training_data)
    >>> prediction = agent.classify("I love this product!")
    >>> print(prediction)  # "positive"

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
from typing import List, Dict, Tuple
from collections import Counter

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


class ClassificationAgent:
    """
    An AI agent that performs text classification using Gemini API with few-shot learning.
    
    This agent uses in-context learning (few-shot prompting) to classify text into
    predefined categories. It demonstrates how LLMs can learn from examples without
    traditional training or fine-tuning.
    
    Attributes:
        model (genai.GenerativeModel): The Gemini model instance.
        training_examples (List[Dict]): Stored examples for few-shot prompting.
        trained (bool): Whether the agent has been "trained" (examples loaded).
    
    Note for Students:
        - Few-shot learning: Model learns from examples in the prompt (no weight updates!)
        - More examples generally improve accuracy but increase API costs
        - Example quality matters more than quantity
        - This approach works well for simple classification tasks
    """
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the classification agent with a specified Gemini model.
        
        Args:
            model_name (str): Name of the Gemini model to use. Defaults to "gemini-3-flash-preview".
        
        Example:
            >>> agent = ClassificationAgent()
            >>> agent_pro = ClassificationAgent("gemini-2.5-pro")
        """
        self.model = genai.GenerativeModel(model_name)
        self.training_examples = []
        self.trained = False
        print(f"✓ Initialized ClassificationAgent with {model_name}")
    
    def train(self, training_data: List[Dict[str, str]]) -> None:
        """
        Train the agent by storing labeled examples for few-shot prompting.
        
        Note: This doesn't actually train the model (no weight updates). It simply stores
        examples that will be included in prompts to guide the model's responses.
        
        Args:
            training_data (List[Dict[str, str]]): List of dicts with 'text' and 'label' keys.
                Example: [{"text": "Great product!", "label": "positive"}, ...]
        
        Example:
            >>> training_data = [
            ...     {"text": "I love it!", "label": "positive"},
            ...     {"text": "Terrible experience", "label": "negative"}
            ... ]
            >>> agent.train(training_data)
        
        Note for Students:
            - Traditional ML: Training updates model weights
            - Few-shot learning: "Training" just stores examples to show the model
            - This is faster and simpler but may be less accurate for complex tasks
        """
        self.training_examples = training_data
        self.trained = True
        print(f"✓ Training complete!")
        print(f"  Stored {len(training_data)} examples for few-shot learning\n")
    
    def classify(self, text: str) -> str:
        """
        Classify a text into positive, negative, or neutral sentiment.
        
        Creates a few-shot prompt with training examples, then asks Gemini to classify
        the input text. Demonstrates practical prompt engineering for classification.
        
        Args:
            text (str): The input text to classify.
            
        Returns:
            str: Predicted label ('positive', 'negative', or 'neutral').
        
        Raises:
            ValueError: If called before train() is called.
        
        Example:
            >>> prediction = agent.classify("This is amazing!")
            >>> print(prediction)  # "positive"
        
        Note for Students:
            - The prompt includes 6 examples (configurable) to guide the model
            - Prompt explicitly requests a single-word response
            - Fallback logic handles cases where model doesn't follow instructions
            - Error handling returns 'neutral' as a safe default
        """
        if not self.trained:
            raise ValueError("Agent must be trained before classification!")
        
        # Create few-shot prompt with training examples
        prompt = "Classify the sentiment of the following text as 'positive', 'negative', or 'neutral'.\n\n"
        prompt += "Examples:\n"
        
        # Add a few training examples
        for i, example in enumerate(self.training_examples[:6]):
            prompt += f"{i+1}. Text: \"{example['text']}\"\n"
            prompt += f"   Sentiment: {example['label']}\n\n"
        
        prompt += f"Now classify this text:\nText: \"{text}\"\n"
        prompt += "Sentiment (respond with only one word: positive, negative, or neutral):"
        
        try:
            response = self.model.generate_content(prompt)
            result = response.text.strip().lower()
            
            # Ensure valid response
            if result in ['positive', 'negative', 'neutral']:
                return result
            else:
                # Fallback: try to extract the sentiment from response
                for sentiment in ['positive', 'negative', 'neutral']:
                    if sentiment in result:
                        return sentiment
                return 'neutral'  # default fallback
        except Exception as e:
            print(f"Error classifying text: {e}")
            return 'neutral'
    
    def classify_batch(self, texts: List[str]) -> List[str]:
        """
        Classify multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of predicted labels
        """
        return [self.classify(text) for text in texts]


class ClassificationEvaluator:
    """Evaluates classification performance using accuracy and F1-score."""
    
    @staticmethod
    def calculate_accuracy(y_true: List[str], y_pred: List[str]) -> float:
        """
        Calculate classification accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Accuracy score (0-1)
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Number of true and predicted labels must match!")
        
        correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
        return round(correct / len(y_true), 4)
    
    @staticmethod
    def calculate_f1_score(y_true: List[str], y_pred: List[str], average: str = 'macro') -> float:
        """
        Calculate F1-score for classification.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            average: Averaging method ('macro' or 'micro')
            
        Returns:
            F1-score (0-1)
        """
        labels = list(set(y_true + y_pred))
        
        if average == 'macro':
            f1_scores = []
            for label in labels:
                precision, recall = ClassificationEvaluator._calculate_precision_recall(
                    y_true, y_pred, label
                )
                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0
                f1_scores.append(f1)
            return round(sum(f1_scores) / len(f1_scores), 4)
        else:
            # Micro averaging
            total_tp = total_fp = total_fn = 0
            for label in labels:
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
                total_tp += tp
                total_fp += fp
                total_fn += fn
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            
            if precision + recall > 0:
                return round(2 * (precision * recall) / (precision + recall), 4)
            else:
                return 0.0
    
    @staticmethod
    def _calculate_precision_recall(y_true: List[str], y_pred: List[str], 
                                   target_label: str) -> Tuple[float, float]:
        """Calculate precision and recall for a specific label."""
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == target_label and p == target_label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != target_label and p == target_label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == target_label and p != target_label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return precision, recall
    
    @staticmethod
    def print_confusion_matrix(y_true: List[str], y_pred: List[str]) -> None:
        """Print a confusion matrix."""
        labels = sorted(set(y_true + y_pred))
        
        # Create matrix
        matrix = {label: {l: 0 for l in labels} for label in labels}
        for true, pred in zip(y_true, y_pred):
            matrix[true][pred] += 1
        
        # Print matrix
        print("\n📊 CONFUSION MATRIX:")
        header_label = "True \\ Pred"
        print(f"\n{header_label:<15}", end='')
        for label in labels:
            print(f"{label:<12}", end='')
        print()
        print("─" * (15 + 12 * len(labels)))
        
        for true_label in labels:
            print(f"{true_label:<15}", end='')
            for pred_label in labels:
                count = matrix[true_label][pred_label]
                print(f"{count:<12}", end='')
            print()
        print()


def main():
    """Main function to demonstrate classification and evaluation."""
    
    # Load sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'sample_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Initialize agent and evaluator
    agent = ClassificationAgent()
    evaluator = ClassificationEvaluator()
    
    print("=" * 80)
    print("TEXT CLASSIFICATION AGENT - DEMONSTRATION")
    print("=" * 80)
    print()
    
    # Train the agent
    print("Training the classification agent...")
    agent.train(data['training_data'])
    
    # Test the agent
    print("Testing the agent on test data...")
    print("─" * 80)
    
    test_texts = [item['text'] for item in data['test_data']]
    test_labels = [item['label'] for item in data['test_data']]
    
    predictions = agent.classify_batch(test_texts)
    
    # Display predictions
    for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, predictions), 1):
        match = "✓" if true_label == pred_label else "✗"
        print(f"\n{i}. Text: {text}")
        print(f"   True Label: {true_label}")
        print(f"   Predicted:  {pred_label} {match}")
    
    # Calculate metrics
    accuracy = evaluator.calculate_accuracy(test_labels, predictions)
    f1_macro = evaluator.calculate_f1_score(test_labels, predictions, average='macro')
    f1_micro = evaluator.calculate_f1_score(test_labels, predictions, average='micro')
    
    print("\n" + "=" * 80)
    print("EVALUATION METRICS:")
    print("─" * 80)
    print(f"  Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  F1-Score (Macro): {f1_macro:.4f}")
    print(f"  F1-Score (Micro): {f1_micro:.4f}")
    
    # Print confusion matrix
    evaluator.print_confusion_matrix(test_labels, predictions)
    
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
