# AI Agent Basics - ML Tasks Tutorial

This educational module covers the fundamentals of AI Agents through practical Machine Learning tasks. Each task demonstrates how agents process data, make decisions, and can be evaluated using established metrics.

## 📚 Overview

This tutorial covers four fundamental ML tasks that AI agents commonly perform:

1. **Text Summarization** - Condensing long texts into concise summaries
2. **Text Classification** - Categorizing text into predefined labels  
3. **Entity Extraction** - Identifying and extracting named entities from text
4. **Structured Output** - Using function calling for reliable, structured data extraction

Each task includes:
- ✅ A complete Python implementation
- ✅ Sample data in JSON format
- ✅ Evaluation metrics (1-2 per task)
- ✅ Clear documentation and examples

## 🗂️ Folder Structure

```
ml_agent_basics/
├── README.md                    # This file
├── summarization/
│   ├── summarization_agent.py   # Summarization agent implementation
│   └── sample_data.json         # Sample texts with reference summaries
├── classification/
│   ├── classification_agent.py  # Classification agent implementation
│   └── sample_data.json         # Training and test data for sentiment analysis
├── entity_extraction/
│   ├── entity_extraction_agent.py  # Entity extraction agent implementation
│   └── sample_data.json            # Texts with annotated entities
└── structured_output/
    ├── structured_output_agent.py  # Structured output with function calling
    └── sample_data.json            # Product, contact, and event data
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7 or higher
- No external dependencies required (uses only Python standard library)

### Running the Examples

Each task can be run independently. Navigate to the respective folder and execute the script:

```bash
# 1. Text Summarization
cd summarization
python summarization_agent.py

# 2. Text Classification
cd ../classification
python classification_agent.py

# 3. Entity Extraction
cd ../entity_extraction
python entity_extraction_agent.py

# 4. Structured Output
cd ../structured_output
python structured_output_agent.py
```

## 📖 Task Descriptions

### 1. Text Summarization 📝

**Goal**: Generate concise summaries from longer texts.

**Agent Approach**: Extractive summarization using keyword scoring and sentence selection.

**Evaluation Metrics**:
- **ROUGE-1**: Measures unigram overlap between generated and reference summaries
  - Precision: % of words in generated summary that appear in reference
  - Recall: % of words in reference summary that appear in generated
  - F1: Harmonic mean of precision and recall
- **BLEU**: Measures n-gram overlap with length penalty

**Sample Output**:
```
ROUGE-1 F1: 0.4500
BLEU Score: 0.3200
```

**How It Works**:
1. Splits text into sentences
2. Scores sentences based on keyword presence
3. Selects top-scoring sentences
4. Compares with reference summary using metrics

---

### 2. Text Classification 🏷️

**Goal**: Categorize text into predefined classes (positive, negative, neutral sentiment).

**Agent Approach**: Keyword-based classification with training on labeled examples.

**Evaluation Metrics**:
- **Accuracy**: Percentage of correct predictions
  - Formula: `correct_predictions / total_predictions`
- **F1-Score**: Harmonic mean of precision and recall
  - Macro: Average F1 across all classes (treats all classes equally)
  - Micro: Global average (weighted by support)

**Sample Output**:
```
Accuracy: 0.8333 (83.33%)
F1-Score (Macro): 0.8250
```

**How It Works**:
1. Trains on labeled examples by extracting class-specific keywords
2. Classifies new text by counting keyword matches
3. Evaluates using accuracy and F1-score
4. Displays confusion matrix for detailed analysis

---

### 3. Entity Extraction 🔍

**Goal**: Identify and extract named entities (persons, organizations, locations, dates) from text.

**Agent Approach**: Pattern-based Named Entity Recognition (NER) using rules and indicators.

**Evaluation Metrics**:
- **Precision**: % of predicted entities that are correct
  - Formula: `true_positives / (true_positives + false_positives)`
- **Recall**: % of actual entities that were found
  - Formula: `true_positives / (true_positives + false_negatives)`
- **F1-Score**: Harmonic mean of precision and recall
  - Calculated overall and per entity type

**Sample Output**:
```
Overall:
  Precision: 0.8571
  Recall: 0.7500
  F1-Score: 0.8000

Per-Type:
  PERSON: F1 = 0.8667
  ORGANIZATION: F1 = 0.8000
  LOCATION: F1 = 0.7500
  DATE: F1 = 1.0000
```

**How It Works**:
1. Uses patterns to identify different entity types
2. Applies rules for organizations, persons, locations, and dates
3. Removes overlapping entities
4. Evaluates with precision, recall, and F1-score

---

### 4. Structured Output 🎯

**Goal**: Extract structured, schema-validated data using Gemini's function calling API.

**Agent Approach**: Define function schemas and let Gemini populate structured data fields.

**Function Types**:
- **Product Extraction**: name, price, currency, storage, color, features
- **Contact Extraction**: name, email, phone, company, title, address
- **Event Extraction**: title, date, time, location, attendees, description

**Sample Output**:
```
📦 Product: iPhone 15 Pro Max
💰 Price: USD1199.00
💾 Storage: 256GB
🎨 Color: Natural Titanium
✨ Features: titanium design, A17 Pro chip, 48MP camera
```

**How It Works**:
1. Defines JSON schemas for each function (product, contact, event)
2. Gemini analyzes text and populates schema fields
3. Returns structured data (no text parsing needed!)
4. Validates that outputs match expected schema

**Key Advantages**:
- ✅ **Reliability**: Schema validation ensures consistent structure
- ✅ **Type Safety**: Numbers are numbers, arrays are arrays
- ✅ **No Parsing**: Direct access to structured data
- ✅ **Production Ready**: Perfect for integration with databases and APIs

## 🎯 Key Concepts for Students

### What is an AI Agent?

An AI agent is a software system that:
- **Perceives** its environment (input data)
- **Processes** information using algorithms
- **Acts** to achieve specific goals (outputs)
- **Learns** from data and feedback

### Why These Tasks?

These four tasks represent fundamental capabilities:
- **Summarization**: Information compression and understanding
- **Classification**: Decision making and categorization
- **Entity Extraction**: Information extraction and structuring
- **Structured Output**: Reliable data extraction for production systems

### Understanding Evaluation Metrics

**Why Metrics Matter**:
- Measure agent performance objectively
- Compare different approaches
- Identify areas for improvement
- Ensure reliability in production

**Choosing the Right Metric**:
- **Accuracy**: Good for balanced datasets
- **F1-Score**: Better for imbalanced classes
- **ROUGE/BLEU**: Standard for text generation
- **Precision vs Recall**: Trade-off based on use case

## 💡 Teaching Tips

1. **Start Simple**: Run each script as-is to see the output
2. **Modify Data**: Edit `sample_data.json` files to test different scenarios
3. **Experiment**: Change agent parameters (e.g., `max_sentences` in summarization)
4. **Compare Approaches**: Try different algorithms and compare metrics
5. **Visualize Results**: Use the confusion matrix in classification
6. **Discuss Trade-offs**: Talk about precision vs recall in entity extraction

## 🔧 Extending the Examples

Students can enhance these agents by:

- **Summarization**:
  - Implement different sentence scoring methods
  - Add abstractive summarization
  - Handle multiple languages

- **Classification**:
  - Add more sentiment classes (e.g., very positive, very negative)
  - Implement TF-IDF weighting
  - Add support for multi-label classification

- **Entity Extraction**:
  - Add more entity types (email, phone, product names)
  - Improve pattern matching rules
  - Add context-aware entity disambiguation

- **Structured Output**:
  - Add more function schemas (recipes, invoices, resumes)
  - Implement validation for extracted data
  - Handle edge cases (missing fields, malformed data)

## 📊 Expected Learning Outcomes

After completing this tutorial, students should understand:

1. **Agent Architecture**: How AI agents process input and generate output
2. **ML Task Types**: Differences between summarization, classification, and extraction
3. **Evaluation**: How to measure and interpret agent performance
4. **Trade-offs**: Balancing precision, recall, and computational efficiency
5. **Practical Application**: How these tasks apply to real-world problems

## 🎓 Workshop Integration

This module serves as the foundation for the "Securing Codebase with ADK and A2A" workshop by:

1. **Establishing Basics**: Understanding what agents do before securing them
2. **Metrics-Driven Development**: Using evaluation to ensure agent reliability
3. **Data Handling**: Working with structured input/output data
4. **Code Quality**: Clean, well-documented agent implementations

## 📝 Questions for Discussion

1. How does the choice of evaluation metric affect agent development?
2. What are the limitations of keyword-based approaches?
3. How would you handle ambiguous cases in entity extraction?
4. Why might a high accuracy agent still fail in production?
5. How can we make these agents more robust and secure?

## 🚀 Next Steps

After mastering these basics:
- Explore deep learning approaches (transformers, BERT, GPT)
- Learn about prompt engineering for LLM-based agents
- Study adversarial attacks on ML models
- Implement security measures for production agents
- Understand model deployment and monitoring

---

**Happy Learning! 🎉**

For questions or issues, please refer to the workshop materials or contact your instructor.
