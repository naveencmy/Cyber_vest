"""
Text Summarization Agent using Gemini API

This module demonstrates an AI agent for text summarization with evaluation metrics.
It showcases how to use the Gemini API for natural language processing tasks and
how to evaluate the quality of generated summaries using industry-standard metrics.

Educational Topics Covered:
- AI Agent Architecture: Input → Processing → Output
- Prompt Engineering: Crafting effective prompts for Gemini API
- Text Summarization: Abstractive vs Extractive approaches
- Evaluation Metrics: ROUGE and BLEU for summary quality assessment
- Error Handling: Graceful degradation when API calls fail

Classes:
    SummarizationAgent: AI agent that generates text summaries using Gemini API
    SummarizationEvaluator: Evaluator that calculates ROUGE and BLEU metrics

Example Usage:
    >>> agent = SummarizationAgent()
    >>> summary = agent.summarize("Your long text here...")
    >>> evaluator = SummarizationEvaluator()
    >>> scores = evaluator.calculate_rouge_1(reference, summary)
    >>> print(f"ROUGE-1 F1: {scores['f1']}")

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
from typing import List, Dict

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


class SummarizationAgent:
    """
    An AI agent that performs text summarization using Gemini API.
    
    This agent uses Google's Gemini model to generate concise summaries of input text.
    It demonstrates practical application of Large Language Models (LLMs) for NLP tasks
    and serves as an educational example of prompt engineering.
    
    Attributes:
        model (genai.GenerativeModel): The Gemini model instance used for text generation.
    
    Note for Students:
        - This agent uses abstractive summarization (generates new sentences)
        - Alternative: Extractive summarization (selects existing sentences)
        - Prompt engineering is crucial: small changes can significantly affect output quality
        - Always handle API errors gracefully in production systems
    """
    
    def __init__(self, model_name: str = "gemini-3-flash-preview"):
        """
        Initialize the summarization agent with a specified Gemini model.
        
        Args:
            model_name (str): Name of the Gemini model to use. Defaults to "gemini-3-flash-preview".
                             Available models include: gemini-2.5-flash, gemini-2.5-pro, etc.
        
        Raises:
            ValueError: If the model_name does not exist or is not accessible.
        
        Example:
            >>> agent = SummarizationAgent()  # Uses default model
            >>> agent_pro = SummarizationAgent("gemini-2.5-pro")  # Uses Pro model
        """
        self.model = genai.GenerativeModel(model_name)
        print(f"✓ Initialized SummarizationAgent with {model_name}")
    
    def summarize(self, text: str) -> str:
        """
        Generate a concise summary of the input text using Gemini API.
        
        The agent creates a prompt instructing Gemini to summarize the text in 1-2 sentences,
       focusing on main ideas and key points. This demonstrates prompt engineering best practices.
        
        Args:
            text (str): The input text to be summarized. Can be any length, but very long texts
                       may hit API token limits or incur higher costs.
        
        Returns:
            str: A concise summary (1-2 sentences) of the input text. Returns an error message
                 if the API call fails.
        
        Raises:
            Exception: Any API errors are caught and logged, returning a safe error message.
        
        Example:
            >>> text = "Artificial Intelligence is transforming industries..."
            >>> summary = agent.summarize(text)
            >>> print(summary)
            "AI is revolutionizing various sectors through automation and data analysis."
        
        Note for Students:
            - The prompt is crucial: it specifies output length, format, and focus
            - Try modifying the prompt to see how it affects the summary quality
            - Error handling prevents the entire script from crashing on API failures
        """
        prompt = f"""Summarize the following text in 1-2 concise sentences. 
Focus on the main ideas and key points.

Text:
{text}

Summary:"""
        
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Error generating summary."


class SummarizationEvaluator:
    """
    Evaluates summarization quality using ROUGE and BLEU metrics.
    
    This class implements two standard metrics for evaluating text summarization:
    - ROUGE-1: Measures unigram (single word) overlap
    - BLEU: Measures n-gram overlap with brevity penalty
    
    These metrics compare generated summaries against reference (human-written) summaries
    to quantify quality objectively.
    
    Note for Students:
        - Higher scores don't always mean better summaries (semantic meaning matters too)
        - ROUGE emphasizes recall (capturing reference content)
        - BLEU emphasizes precision (avoiding extra content)
        - In practice, use multiple metrics to get a complete picture
    """
    
    @staticmethod
    def calculate_rouge_1(reference: str, candidate: str) -> Dict[str, float]:
        """
        Calculate ROUGE-1 (Recall-Oriented Understudy for Gisting Evaluation) scores.
        
        ROUGE-1 measures the overlap of unigrams (individual words) between the reference
        and candidate summaries. It's one of the most widely used metrics for evaluating
        automatic summarization.
        
        Args:
            reference (str): The reference (gold standard) summary, typically human-written.
            candidate (str): The generated summary to evaluate.
            
        Returns:
            Dict[str, float]: Dictionary containing:
                - 'precision': Fraction of candidate words that appear in reference
                - 'recall': Fraction of reference words captured in candidate
                - 'f1': Harmonic mean of precision and recall (balanced metric)
        
        Example:
            >>> ref = "AI transforms industries"
            >>> can = "AI is transforming many industries"
            >>> scores = SummarizationEvaluator.calculate_rouge_1(ref, can)
            >>> print(f"F1: {scores['f1']}")  # ~0.67
        
        Note for Students:
            - Precision = How many words in summary are relevant?
            - Recall = How much of the reference is covered?
            - F1 balances both (useful when precision & recall trade off)
            - Case-insensitive comparison (converts to lowercase)
        """
        ref_tokens = set(reference.lower().split())
        cand_tokens = set(candidate.lower().split())
        
        if len(cand_tokens) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        overlap = len(ref_tokens & cand_tokens)
        
        precision = overlap / len(cand_tokens) if len(cand_tokens) > 0 else 0
        recall = overlap / len(ref_tokens) if len(ref_tokens) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4)
        }
    
    @staticmethod
    def calculate_bleu(reference: str, candidate: str, n: int = 2) -> float:
        """
        Calculate simplified BLEU (Bilingual Evaluation Understudy) score.
        
        BLEU measures the quality of machine-generated text by comparing n-gram overlap
        with reference text. Originally designed for machine translation, it's also
        useful for summarization evaluation.
        
        Args:
            reference (str): The reference summary.
            candidate (str): The generated summary to evaluate.
            n (int): Maximum n-gram size to consider (default 2 for unigrams and bigrams).
                    Higher n captures longer phrase matching.
            
        Returns:
            float: BLEU score between 0.0 and 1.0, where higher is better.
                  Includes brevity penalty to penalize overly short candidates.
        
        Example:
            >>> ref = "The cat sat on the mat"
            >>> can = "A cat sat on a mat"
            >>> score = SummarizationEvaluator.calculate_bleu(ref, can)
            >>> print(f"BLEU: {score}")  # ~0.45 (good partial match)
        
        Note for Students:
            - N-grams: sequences of n consecutive words (1-gram="cat", 2-gram="cat sat")
            - Brevity penalty: Prevents gaming the metric with very short outputs
            - This is a simplified version; production systems use smoothed BLEU
            - BLEU complements ROUGE: BLEU focuses on precision, ROUGE on recall
        """
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        
        if len(cand_tokens) == 0:
            return 0.0
        
        # Calculate n-gram precision
        precisions = []
        for i in range(1, n + 1):
            ref_ngrams = SummarizationEvaluator._get_ngrams(ref_tokens, i)
            cand_ngrams = SummarizationEvaluator._get_ngrams(cand_tokens, i)
            
            if len(cand_ngrams) == 0:
                continue
            
            # Calculate overlap
            overlap = 0
            for ngram in cand_ngrams:
                if ngram in ref_ngrams:
                    overlap += min(ref_ngrams[ngram], cand_ngrams[ngram])
            precision = overlap / sum(cand_ngrams.values())
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        bleu = (sum(precisions) / len(precisions))
        
        # Length penalty
        brevity_penalty = min(1.0, len(cand_tokens) / len(ref_tokens)) if len(ref_tokens) > 0 else 0
        
        return round(bleu * brevity_penalty, 4)
    
    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Dict[tuple, int]:
        """
        Generate n-grams from a list of tokens.
        
        Args:
            tokens (List[str]): List of word tokens.
            n (int): Size of n-grams to generate.
        
        Returns:
            Dict[tuple, int]: Dictionary mapping n-gram tuples to their frequency counts.
        
        Example:
            >>> tokens = ["the", "cat", "sat"]
            >>> bigrams = SummarizationEvaluator._get_ngrams(tokens, 2)
            >>> print(bigrams)  # {('the', 'cat'): 1, ('cat', 'sat'): 1}
        """
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i + n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams


def main():
    """Main function to demonstrate summarization and evaluation."""
    
    # Load sample data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, 'sample_data.json')
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Initialize agent and evaluator
    agent = SummarizationAgent()
    evaluator = SummarizationEvaluator()
    
    print("=" * 80)
    print("TEXT SUMMARIZATION AGENT - DEMONSTRATION")
    print("=" * 80)
    
    total_rouge_f1 = 0
    total_bleu = 0
    
    for item in data['texts']:
        print(f"\n{'─' * 80}")
        print(f"Text ID: {item['id']}")
        print(f"{'─' * 80}")
        print(f"\nOriginal Text ({len(item['text'])} chars):")
        print(item['text'])
        
        # Generate summary
        summary = agent.summarize(item['text'])
        print(f"\nGenerated Summary ({len(summary)} chars):")
        print(summary)
        
        print(f"\nReference Summary:")
        print(item['reference_summary'])
        
        # Evaluate
        rouge_scores = evaluator.calculate_rouge_1(item['reference_summary'], summary)
        bleu_score = evaluator.calculate_bleu(item['reference_summary'], summary)
        
        print(f"\n📊 EVALUATION METRICS:")
        print(f"  ROUGE-1 Precision: {rouge_scores['precision']:.4f}")
        print(f"  ROUGE-1 Recall:    {rouge_scores['recall']:.4f}")
        print(f"  ROUGE-1 F1:        {rouge_scores['f1']:.4f}")
        print(f"  BLEU Score:        {bleu_score:.4f}")
        
        total_rouge_f1 += rouge_scores['f1']
        total_bleu += bleu_score
    
    # Average metrics
    num_samples = len(data['texts'])
    print(f"\n{'=' * 80}")
    print(f"AVERAGE METRICS ACROSS {num_samples} SAMPLES:")
    print(f"  Average ROUGE-1 F1: {total_rouge_f1 / num_samples:.4f}")
    print(f"  Average BLEU Score: {total_bleu / num_samples:.4f}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
