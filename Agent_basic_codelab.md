summary: AI Agent Basics with Gemini - ML Tasks Tutorial (60 min)
id: gdg-ai-agent-basics-ml-60
categories: AI, Machine Learning, Gemini
tags: gemini, ai-agents, ml, python, function-calling
status: Published
authors: Aditya Mahakali ( [LinkedIn](https://www.linkedin.com/in/aditya-mahakali-b81758168/))
Feedback Link: https://github.com/GDG-Bangalore/GDG_workshop_Securing_Codebase_with_ADK_and_A2A/issues

# AI Agent Basics with Gemini - ML Tasks Tutorial (60 min)

<!-- ------------------------ -->
## 1. Overview

Duration: 5

In this codelab you will learn **AI Agent fundamentals** through 4 practical ML tasks using:

- **Gemini API** – Google's powerful LLM for text generation
- **Python** – clean, educational code examples
- **Function Calling** – structured output for production systems
- **Evaluation Metrics** – ROUGE, BLEU, Accuracy, F1-Score

The 4 ML Tasks:

1. **Text Summarization** → condense long texts into summaries (ROUGE, BLEU metrics)
2. **Text Classification** → sentiment analysis with few-shot learning (Accuracy, F1)
3. **Entity Extraction** → identify persons, organizations, locations (Precision, Recall)
4. **Structured Output** → use function calling for reliable data extraction

You'll:

- Run 4 complete AI agents locally
- Understand evaluation metrics
- Learn prompt engineering techniques
- See production-ready patterns (function calling)

---

<!-- ------------------------ -->
## 2. Prerequisites & Repo Setup

Duration: 5

### 2.1. What you need

- **Python** 3.7 or higher (3.10+ recommended)
- **Git** installed
- A **Gemini API key** (free tier available)

### 2.2. Get a Gemini API key

If you don't already have one:

1. Go to `https://ai.google.dev`
2. Sign in with your Google account
3. Create an API key
4. Copy it (you'll paste it into `.env`)

### 2.3. Clone the workshop repo

In a terminal:

```bash
git clone https://github.com/GDG-Bangalore/GDG_workshop_Securing_Codebase_with_ADK_and_A2A.git
cd GDG_workshop_Securing_Codebase_with_ADK_and_A2A/ml_agent_basics
```

You should see:

```text
ml_agent_basics/
├── README.md
├── summarization/
├── classification/
├── entity_extraction/
└── structured_output/
```

---

<!-- ------------------------ -->
## 3. Virtualenv & Dependencies

Duration: 5

To keep things clean, use a virtual environment.

### 3.1. Create venv

```bash
cd /path/to/GDG_workshop_Securing_Codebase_with_ADK_and_A2A
python3 -m venv venv
```

### 3.2. Activate venv

**macOS / Linux:**

```bash
source venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

Make sure you see `(venv)` at the start of your terminal prompt.

### 3.3. Install dependencies

```bash
pip install google-generativeai python-dotenv
```

These are the only two packages needed for all 4 agents!

---

<!-- ------------------------ -->
## 4. Configure `.env` (Gemini API Key)

Duration: 3

All agents use `python-dotenv` to load your API key.

### 4.1. Create `.env` file

In the project root (parent of `ml_agent_basics/`):

```bash
touch .env
```

### 4.2. Edit `.env`

Open `.env` and add your Gemini API key:

```env
GOOGLE_API_KEY="YOUR_REAL_GEMINI_API_KEY"
```

Alternatively, you can use:

```env
GEMINI_API_KEY="YOUR_REAL_GEMINI_API_KEY"
```

Save the file. All agents will automatically load this key.

---

<!-- ------------------------ -->
## 5. Task 1: Text Summarization

Duration: 10

**Goal**: Generate concise summaries from longer texts using Gemini.

### 5.1. Run the summarization agent

```bash
cd ml_agent_basics/summarization
python summarization_agent.py
```

### 5.2. What you'll see

```text
 Initialized SummarizationAgent with gemini-3-flash-preview

================================================================================
TEXT SUMMARIZATION AGENT - DEMONSTRATION
================================================================================

Text 1:
  Original: "Artificial Intelligence is rapidly transforming industries..."
  Summary: "AI is revolutionizing various sectors through automation."

 METRICS:
  ROUGE-1 Precision: 0.4000
  ROUGE-1 Recall: 0.3333
  ROUGE-1 F1: 0.3636
  BLEU Score: 0.2500
```

### 5.3. Understanding the metrics

- **ROUGE-1**: Measures word overlap (unigrams)
  - **Precision**: % of summary words in reference
  - **Recall**: % of reference words in summary
  - **F1**: Harmonic mean (balanced score)

- **BLEU**: Measures n-gram overlap with length penalty
  - Higher BLEU = better phrase matching

### 5.4. Educational notes

Open `summarization_agent.py` and look for:
- `SummarizationAgent.summarize()` - see the prompt engineering
- `SummarizationEvaluator.calculate_rouge_1()` - understand the math
- Docstrings with "Note for Students" sections

---

<!-- ------------------------ -->
## 6. Task 2: Text Classification

Duration: 10

**Goal**: Classify text sentiment using few-shot learning (no training needed!).

### 6.1. Run the classification agent

```bash
cd ../classification
python classification_agent.py
```

### 6.2. What you'll see

```text
 Initialized ClassificationAgent with gemini-3-flash-preview
 Training complete!
  Stored 15 examples for few-shot learning

Testing the agent on test data...

1. Text: "This is absolutely amazing!"
   True Label: positive
   Predicted:  positive 

EVALUATION METRICS:
  Accuracy:        1.0000 (100.00%)
  F1-Score (Macro): 1.0000
  F1-Score (Micro): 1.0000

 CONFUSION MATRIX:
...
```

### 6.3. Understanding few-shot learning

- **No weight updates**: The model doesn't "train" in the traditional sense
- **In-context learning**: Examples are included in the prompt
- **6 examples** are shown to guide the model's responses

### 6.4. Understanding the metrics

- **Accuracy**: `correct_predictions / total_predictions`
- **F1-Score**:
  - **Macro**: Average per class (treats all classes equally)
  - **Micro**: Global average (weighted by support)
- **Confusion Matrix**: Shows which classes were confused

### 6.5. Experiment

Try modifying `sample_data.json`:
- Add more test examples
- Change sentiments and see how it performs
- Add more training examples for better accuracy

---

<!-- ------------------------ -->
## 7. Task 3: Entity Extraction

Duration: 10

**Goal**: Extract named entities (PERSON, ORGANIZATION, LOCATION, DATE) from text.

### 7.1. Run the entity extraction agent

```bash
cd ../entity_extraction
python entity_extraction_agent.py
```

### 7.2. What you'll see

```text
 Initialized EntityExtractionAgent with gemini-3-flash-preview

================================================================================
ENTITY EXTRACTION AGENT - DEMONSTRATION
================================================================================

Text ID: 1
────────────────────────────────────────────────────────────────────
Text: "Apple Inc. CEO Tim Cook announced new products in Cupertino on March 15, 2024."

Ground Truth Entities:
  - Apple Inc.                  [ORGANIZATION]
  - Tim Cook                    [PERSON]
  - Cupertino                   [LOCATION]
  - March 15, 2024              [DATE]

Predicted Entities:
  - Apple Inc.                  [ORGANIZATION]
  - Tim Cook                    [PERSON]
  - Cupertino                   [LOCATION]
  - March 15, 2024              [DATE]

 METRICS FOR THIS TEXT:
  Precision: 1.0000
  Recall:    1.0000
  F1-Score:  1.0000
```

### 7.3. Understanding the metrics

- **Precision**: `true_positives / (true_positives + false_positives)`
  - How many extracted entities are correct?
- **Recall**: `true_positives / (true_positives + false_negatives)`
  - How many real entities did we find?
- **F1-Score**: Harmonic mean of precision and recall

### 7.4. Structured output

Notice that entities include:
- `text`: The entity text
- `type`: Entity category
- `start`, `end`: Character positions for highlighting in UI

This is **structured JSON output** - no regex parsing needed!

---

<!-- ------------------------ -->
## 8. Task 4: Structured Output (Function Calling)

Duration: 15

**Goal**: Use Gemini's function calling API for reliable, schema-validated data extraction.

### 8.1. Why function calling?

Traditional approach:
```python
#  Parse free-form text (error-prone)
response = model.prompt("Extract product info...")
price = extract_price_with_regex(response.text)  # fragile!
```

Function calling approach:
```python
#  Get structured data directly (reliable)
response = model.call_function("extract_product_info", ...)
price = response.args["price"]  # type-safe!
```

### 8.2. Run the structured output agent

```bash
cd ../structured_output
python structured_output_agent.py
```

### 8.3. What you'll see

```text
 Initialized StructuredOutputAgent with gemini-3-flash-preview
 Loaded 3 function schemas

================================================================================
STRUCTURED OUTPUT AGENT - DEMONSTRATION
================================================================================

────────────────────────────────────────────────────────────────────
 PRODUCT INFORMATION EXTRACTION
────────────────────────────────────────────────────────────────────

Input Text: "The new iPhone 15 Pro Max features a titanium design and costs $1199..."

 Product: iPhone 15 Pro Max
 Price: USD1199.00
 Storage: 256GB
 Color: Natural Titanium
 Features: titanium design, A17 Pro chip, 48MP camera, USB-C port

────────────────────────────────────────────────────────────────────
 CONTACT INFORMATION EXTRACTION
────────────────────────────────────────────────────────────────────

Input Text: "Dr. Sarah Johnson, Chief Technology Officer at TechCorp Inc..."

 Name: Dr. Sarah Johnson
 Email: sarah.johnson@techcorp.com
 Phone: +1 (555) 234-5678
 Company: TechCorp Inc.
 Title: Chief Technology Officer
 Address: 123 Innovation Drive, San Francisco, CA 94102

────────────────────────────────────────────────────────────────────
 EVENT INFORMATION EXTRACTION
────────────────────────────────────────────────────────────────────

Input Text: "Annual Tech Conference scheduled for March 15, 2024 at 9:00 AM..."

 Event: Annual Tech Conference
 Date: 2024-03-15
 Time: 9:00 AM
 Location: Grand Ballroom, Convention Center
 Attendees: Tim Cook, Satya Nadella, Sundar Pichai
 Description: Featuring keynotes from industry leaders. Topics include AI, cloud computing...
```

### 8.4. Understanding function schemas

Open `structured_output_agent.py` and find the `__init__` method. You'll see:

```python
genai.protos.FunctionDeclaration(
    name="extract_product_info",
    description="Extract product information from text...",
    parameters=genai.protos.Schema(
        type=genai.protos.Type.OBJECT,
        properties={
            "name": genai.protos.Schema(type=genai.protos.Type.STRING),
            "price": genai.protos.Schema(type=genai.protos.Type.NUMBER),
            "features": genai.protos.Schema(
                type=genai.protos.Type.ARRAY,
                items=genai.protos.Schema(type=genai.protos.Type.STRING)
            )
        },
        required=["name", "price", "currency"]
    )
)
```

This is **JSON Schema** - a contract between your code and the AI!

### 8.5. Key advantages

 **Type Safety**: Numbers are numbers, not strings  
 **Validation**: Required fields are guaranteed  
 **No Parsing**: Direct dictionary access  
 **Production Ready**: Used in enterprise AI systems

### 8.6. Experiment

Try modifying `sample_data.json`:
- Add your own product descriptions
- Test with contact info from business cards
- Create event invitations

---

<!-- ------------------------ -->
## 9. Code Walkthrough

Duration: 7

Let's highlight the key patterns used across all agents.

### 9.1. Agent structure

Every agent follows this pattern:

```python
class Agent:
    def __init__(self, model_name="gemini-3-flash-preview"):
        self.model = genai.GenerativeModel(model_name)
    
    def main_method(self, input_text):
        prompt = f"... {input_text} ..."
        response = self.model.generate_content(prompt)
        return response.text.strip()
```

### 9.2. Evaluation pattern

Every evaluator provides static methods:

```python
class Evaluator:
    @staticmethod
    def calculate_metrics(true_values, predicted_values):
        # Calculate precision, recall, F1, etc.
        return metrics_dict
```

### 9.3. Docstrings for learning

Look for these sections in the code:
- **Module docstring**: Educational topics covered
- **Class docstring**: Architecture and attributes
- **Method docstring**: Parameters, returns, examples
- **"Note for Students"**: Key learning points

### 9.4. Error handling

All agents implement:
```python
try:
    response = self.model.generate_content(prompt)
    return response.text
except Exception as e:
    print(f"Error: {e}")
    return "Error generating response."
```

This prevents crashes and helps debugging!

---

<!-- ------------------------ -->
## 10. Key Concepts Summary

Duration: 5

### 10.1. What is an AI Agent?

An AI agent:
- **Perceives** input data
- **Processes** using algorithms (LLMs, prompts)
- **Acts** to produce outputs
- **Learns** from feedback (evaluation metrics)

### 10.2. Prompt engineering matters

Different prompts = different quality:

**Bad prompt**:
```
Summarize this
```

**Good prompt**:
```
Summarize the following text in 1-2 concise sentences.
Focus on the main ideas and key points.
```

### 10.3. Evaluation is essential

Metrics help you:
- Measure performance objectively
- Compare different approaches
- Identify areas for improvement
- Ensure reliability before production

### 10.4. Function calling is production-ready

For real applications:
-  Don't parse free-form text with regex
-  Use function calling for structured output
-  Define schemas that match your database/API

---

<!-- ------------------------ -->
## 11. Extending the Examples

Duration: 5

Here are some ideas to enhance these agents after the workshop:

### 11.1. Summarization

- Add a "summarize_to_length" parameter (e.g., 50 words, 100 words)
- Implement **abstractive summarization** (rewrite in own words)
- Compare different Gemini models (Flash vs Pro)

### 11.2. Classification

- Add more classes: "very positive", "very negative"
- Implement **multi-label classification** (multiple labels per text)
- Try **zero-shot classification** (no training examples!)

### 11.3. Entity Extraction

- Add more entity types: EMAIL, PHONE, PRODUCT
- Implement **entity linking** (connect entities to knowledge bases)
- Add **confidence scores** for each entity

### 11.4. Structured Output

- Create function schemas for:
  - **Resume parsing** (name, skills, experience)
  - **Invoice extraction** (items, prices, totals)
  - **Recipe parsing** (ingredients, steps, cook time)
- Chain multiple functions together
- Add validation logic for extracted data

---

<!-- ------------------------ -->
## 12. Common Issues & Solutions

Duration: 3

### Issue: "No Gemini API key found"

**Solution**: Make sure `.env` file exists in project root with:
```env
GOOGLE_API_KEY="your_key_here"
```

### Issue: "404 Model not found"

**Solution**: The model name changed. All scripts now use `gemini-3-flash-preview`. If this fails:
1. Check Google AI Studio for current model names
2. Update the model name in each agent's `__init__` method

### Issue: "Rate limit exceeded (429)"

**Solution**:
- You're making too many API calls too quickly
- Add `time.sleep(2)` between requests
- Use the free tier's quota wisely
- Consider upgrading to paid tier for production

### Issue: Function calling returns None

**Solution**:
- Check that the prompt clearly requests the function
- Verify the function schema is correct
- Look at the raw response to see what Gemini returned

---

<!-- ------------------------ -->
## 13. Where to Go Next

Duration: 3

You now understand **AI Agent fundamentals**:

 How agents process input and generate output  
 Prompt engineering techniques  
 Evaluation metrics (ROUGE, BLEU, Accuracy, F1)  
 Few-shot learning  
 Structured output with function calling

### Next steps:

1. **Explore More Gemini Features**:
   - Multimodal inputs (images + text)
   - Streaming responses
   - System instructions

2. **Learn Advanced Patterns**:
   - Multi-turn conversations (chat history)
   - Agent chains (output of one → input of another)
   - Retrieval Augmented Generation (RAG)

3. **Build Production Systems**:
   - Add authentication and rate limiting
   - Implement caching for common queries
   - Monitor API costs and performance
   - Handle edge cases and errors gracefully

4. **Secure Your Agents**:
   - Continue with the main workshop on ADK, A2A, and security
   - Learn about prompt injection attacks
   - Implement input validation
   - Audit agent behavior

---

## 14. Resources & Links

Duration: 2

### Official Documentation

- [Gemini API Docs](https://ai.google.dev/docs)
- [Function Calling Guide](https://ai.google.dev/docs/function_calling)
- [Python SDK Reference](https://ai.google.dev/api/python)

### Workshop Materials

- [GitHub Repository](https://github.com/GDG-Bangalore/GDG_workshop_Securing_Codebase_with_ADK_and_A2A)
- [Main Codelab](codelab.md) - Multi-agent security scanner
- [Report Issues](https://github.com/GDG-Bangalore/GDG_workshop_Securing_Codebase_with_ADK_and_A2A/issues)

### Learning More

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [ML Evaluation Metrics](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
- [Google AI Studio](https://aistudio.google.com/) - Test prompts visually

---

**Happy Learning! **

You've completed the AI Agent Basics tutorial. Keep experimenting with the code, try the extensions, and most importantly - have fun building with AI agents!
