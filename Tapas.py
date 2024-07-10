from transformers import TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import torch

# Load the tokenizer and model
tokenizer = TapasTokenizer.from_pretrained("google/tapas-large-finetuned-wtq")
model = TapasForQuestionAnswering.from_pretrained("google/tapas-large-finetuned-wtq")

# Create a DataFrame representing the table
data = {
    "Content": ["Chat GPT comes from OpenAI", "Chat GPT is a LLM", "Chat GPT is very quality rich"],
    "Number": ["1", "2", "3"]
}
table = pd.DataFrame(data)

# Define the query
query = "Under which Number do I find the answer to the question: 'What is ChatGPT?'"

# Tokenize the table and query
inputs = tokenizer(table=table, queries=query, padding="max_length", return_tensors="pt")

# Get the model output
with torch.no_grad():
    outputs = model(**inputs)

# Get the predicted answer
predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs,
    outputs.logits.detach(),
    outputs.logits_aggregation.detach()
)

number_column_value = table.iloc[predicted_answer_coordinates[0][0][0]+1]['Number']
print(number_column_value)