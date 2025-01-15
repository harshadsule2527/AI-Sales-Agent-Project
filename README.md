# AI Sales Agent Project
 
# Sales Conversation Fine-Tuning

This repository contains code to fine-tune the `distilgpt2` model from the Transformers library to handle sales-related conversations effectively. The model is trained on synthetic data consisting of common customer inquiries and salesperson responses to provide contextually relevant and customer-focused responses.

## Features
- Fine-tunes a pre-trained GPT-2 model for sales conversations.
- Provides a framework for preparing datasets, fine-tuning, and evaluation.
- Supports GPU acceleration for efficient training.
- Saves fine-tuned models and evaluation results for reuse.

## Getting Started

### Prerequisites
1. Python 3.8 or higher.
2. PyTorch and the Hugging Face Transformers library.
3. CUDA-enabled GPU for faster training (optional).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sales-conversation-finetuning.git
   cd sales-conversation-finetuning
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Dataset
The dataset for fine-tuning consists of synthetic conversations between customers and sales representatives. Each example includes an `input` (customer's question or concern) and an `output` (salesperson's response).

You can add your own dataset by modifying the `conversations` list in the script.

### Training the Model
1. Run the fine-tuning script:
   ```bash
   python fine_tune.py
   ```

2. The fine-tuned model will be saved in the `./fine_tuned_model` directory.

### Evaluating the Model
1. Modify the `evaluation_prompts` list in the script with your desired prompts.
2. Run the evaluation script:
   ```bash
   python fine_tune.py
   ```
3. Evaluation results will be saved in `evaluation_results.txt`.

### Usage
You can use the fine-tuned model to generate sales-focused responses by providing customer prompts. Example:
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")

def generate_response(prompt):
    input_text = f"Input: {prompt}\nOutput:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

prompt = "Customer: How do you ensure the software is scalable?"
response = generate_response(prompt)
print(response)
```

## File Structure
- `fine_tune.py`: Main script for dataset preparation, training, and evaluation.
- `requirements.txt`: Required Python libraries.
- `evaluation_results.txt`: Results of the evaluation.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions and improvements.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
