import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import Dataset

# Create synthetic dataset
conversations = [
    {
        "input": "Customer: I'm interested in your software, but I'm not sure if it's worth the price.",
        "output": "Salesperson: I understand your concern about the price. Let's break down the value proposition. What specific features are most important for your business needs? This way, we can focus on how our software can provide a return on investment in those areas."
    },
    {
        "input": "Customer: Your competitor offers a similar product at a lower price. Why should I choose yours?",
        "output": "Salesperson: That's a great question. While price is certainly a factor, it's important to consider the total value. Could you tell me more about your specific needs? This will help me highlight how our unique features and customer support might justify the price difference and potentially save you money in the long run."
    },
    {
        "input": "Customer: I'm worried about the learning curve for my team. How difficult is it to implement your solution?",
        "output": "Salesperson: Your concern about the learning curve is valid. Let's approach this step-by-step. First, could you tell me about your team's current technical expertise? This will help me explain our onboarding process and how we tailor our training to different skill levels. We also offer ongoing support to ensure a smooth transition."
    },
    {
        "input": "Customer: We've had bad experiences with similar software in the past. How can I trust that yours won't have the same issues?",
        "output": "Salesperson: I'm sorry to hear about your past experiences. To address your concern, could you share what specific issues you encountered? This will help me explain how our software is designed to avoid those problems. Additionally, I can provide case studies of clients who had similar concerns and how we successfully addressed them."
    },
    {
        "input": "Customer: Your software seems complex. I'm not sure if we need all these features.",
        "output": "Salesperson: I appreciate your honesty about the complexity. Let's take a step back and focus on your core needs. What are the main challenges you're trying to solve? We can then discuss which features directly address those needs and how they can be implemented gradually to avoid overwhelming your team."
    },
    {
        "input": "Customer: We're a small business. Is your software scalable as we grow?",
        "output": "Salesperson: Scalability is indeed crucial for growing businesses. Can you share your projected growth plans for the next 2-3 years? This will help me demonstrate how our software can adapt to your changing needs. We offer flexible pricing tiers and modular features that can be added as your business expands."
    },
    {
        "input": "Customer: I'm concerned about data security. How do you ensure our information is protected?",
        "output": "Salesperson: Data security is a top priority for us. To address your concern, could you tell me about your current security protocols? This will help me explain how our security measures can integrate with and enhance your existing practices. I can also provide details on our encryption methods, regular security audits, and compliance certifications."
    },
    {
        "input": "Customer: We're considering developing our own in-house solution. Why should we choose your software instead?",
        "output": "Salesperson: Developing an in-house solution is certainly an option. To help you make an informed decision, could you share your reasons for considering this approach? This will allow me to compare the costs, time-to-market, and long-term maintenance aspects of both options. I can also explain how our software's flexibility might offer a middle ground, allowing for customization without the full burden of in-house development."
    },
    {
        "input": "Customer: Your support hours don't align with our business hours. How can we ensure timely assistance?",
        "output": "Salesperson: I understand your concern about support availability. Can you tell me more about your business hours and the types of support you typically need? This information will help me explore options like extended support hours, dedicated support channels, or even the possibility of aligning a support team with your time zone to ensure you receive timely assistance when you need it most."
    },
    {
        "input": "Customer: We need to integrate this with our existing systems. How compatible is your software?",
        "output": "Salesperson: Integration compatibility is crucial for a seamless workflow. Could you provide details about your current systems and the specific integrations you need? This will allow me to explain our API capabilities, existing integrations, and how we've successfully integrated with similar systems for other clients. If needed, I can also involve our technical team to discuss custom integration solutions."
    }
]

# Prepare the dataset
def prepare_dataset(conversations):
    texts = [f"Input: {conv['input']}\nOutput: {conv['output']}" for conv in conversations]
    dataset = Dataset.from_dict({"text": texts})
    return dataset

dataset = prepare_dataset(conversations)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model and tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Set the pad token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

# Fine-tune the model
print("Starting model fine-tuning...")
trainer.train()

# Save the fine-tuned model
print("Saving the fine-tuned model...")
trainer.save_model("./fine_tuned_model")

# Evaluation function
def evaluate_model(model, tokenizer, prompts, device):
    model.eval()
    results = []

    for prompt in prompts:
        input_text = f"Input: {prompt}\nOutput:"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        
        with torch.no_grad():
            output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2)
        
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})
    
    return results

# Evaluation prompts
evaluation_prompts = [
    "Customer: Your product seems expensive compared to others in the market. Why?",
    "Customer: How long does it typically take to see results after implementing your solution?",
    "Customer: We're worried about employee adoption. How can you help with that?",
    "Customer: What makes your customer service stand out from competitors?",
    "Customer: Can your software handle our large volume of data?"
]

# Perform evaluation
print("Evaluating the fine-tuned model...")
evaluation_results = evaluate_model(model, tokenizer, evaluation_prompts, device)

# Print evaluation results
print("\nEvaluation Results:")
for result in evaluation_results:
    print(f"Prompt: {result['prompt']}")
    print(f"Response: {result['response']}\n")

# Save evaluation results to a file
with open("evaluation_results.txt", "w") as f:
    for result in evaluation_results:
        f.write(f"Prompt: {result['prompt']}\n")
        f.write(f"Response: {result['response']}\n\n")

print("Evaluation results have been saved to 'evaluation_results.txt'")