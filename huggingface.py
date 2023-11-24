import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import warnings
import re
warnings.filterwarnings('ignore')

# Initialize tokenizer and model
model_id = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def is_arithmetic_expression(s):
    """ Check if the string is a simple arithmetic expression """
    return bool(re.match(r"^\s*\d+(\s*[-+*/]\s*\d+)*\s*$", s))

while True:
    # Prompt for user input
    input_text = str(input("Enter your prompt (or type 'exit' to quit): "))

    # Check for the exit command
    if input_text.lower() == 'exit':
        break

    # Check if the input is an arithmetic expression
    if is_arithmetic_expression(input_text):
        try:
            # Safely evaluate the arithmetic expression
            result = eval(input_text)
            print(f"Calculation result: {result}")
        except Exception as e:
            print(f"Error in calculation: {e}")
    else:
        # Generate response from the language model
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids)

        # Format and print the output
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Model output response: {response}")

# Optional: Add any cleanup code here if necessary
