from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# Text parts
text_parts = [
    "1. Chat GPT comes from OpenAI.",
    "2. Chat GPT is very quality rich.",
    "3. Chat GPT is a LLM."
]

# User query
query = "In which paragraph do I find answers to the following question: 'What is ChatGPT?'"

# Combine the text parts for input
combined_text = " ".join(text_parts)

# Tokenize the inputs
inputs = tokenizer(query, combined_text, add_special_tokens=True, return_tensors="pt")

# Get the model output
outputs = model(**inputs)

# Get the start and end logits
start_logits = outputs.start_logits
end_logits = outputs.end_logits

# Find the start and end positions
start_position = torch.argmax(start_logits)
end_position = torch.argmax(end_logits)

# Convert token positions to char positions
input_ids = inputs["input_ids"].squeeze().tolist()
all_tokens = tokenizer.convert_ids_to_tokens(input_ids)
answer_tokens = all_tokens[start_position:end_position+1]
answer = tokenizer.convert_tokens_to_string(answer_tokens)

# Determine which part contains the answer
def find_text_part(answer, text_parts):
    for part in text_parts:
        if answer in part:
            return part.split('.')[0]
    return None

text_part_number = find_text_part(answer, text_parts)

print(f"Output: {text_part_number}")

# Debugging outputs
print(f"Query: {query}")
print(f"Combined Text: {combined_text}")
print(f"Start Position: {start_position}")
print(f"End Position: {end_position}")
print(f"Answer: {answer}")
