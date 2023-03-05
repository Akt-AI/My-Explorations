import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load the Bloom model
tokenizer = AutoTokenizer.from_pretrained("huggingface/bloom-large-transformers-lm")
model = AutoModelWithLMHead.from_pretrained("huggingface/bloom-large-transformers-lm")

# Set the device to run the model on (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the prompt or input text
prompt = "In a world where"

# Encode the prompt and add special tokens to indicate the start and end of the text
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# Generate text
with torch.no_grad():
    output = model.generate(input_ids=input_ids, max_length=50)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)

