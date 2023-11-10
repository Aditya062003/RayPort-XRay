import tkinter as tk
from tkinter import scrolledtext
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load the model and the tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Function to send message
def send_message():
    message = user_input.get()
    chat_window.insert(tk.END, f"You: {message}\n\n")
    user_input.delete(0, tk.END)

    context = """
If a person has a cough and a fever, they might have the flu.
If a person has shortness of breath and chest pain, they might have pneumonia.
If a person has a persistent cough, chest pain, and loss of appetite, they might have lung cancer.
If a person has wheezing and shortness of breath, they might have asthma.
If a person has fatigue, a persistent cough, and difficulty breathing, they might have chronic obstructive pulmonary disease (COPD).
If a person has rapid breathing, chest pain, and coughing up blood, they might have a pulmonary embolism.
If a person has a dry cough, shortness of breath, and fatigue, they might have idiopathic pulmonary fibrosis.
"""

    # Prepare the question
    question = f"What disease might a person have if they have {message}?"

    # Use the model to get an answer
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt")
    outputs = model(**inputs)
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    chat_window.insert(
        tk.END,
        f"Computer: You may be suffering from {answer}. Kindly take professional medical help.\n\n",
    )


# Create the main window
root = tk.Tk()
root.title("Chat Interface")

# Create the chat window
chat_window = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    bg="light grey",
    fg="blue",
    font=("Helvetica", 10),
)
chat_window.grid(row=0, column=0, columnspan=2, sticky="nsew")

# Insert the first message
chat_window.insert(
    tk.END, "Computer: Kindly list out your current respiratory symptoms.\n\n"
)

# Create the user input field
user_input = tk.Entry(root, bg="light grey", fg="black", font=("Arial", 10))
user_input.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)

# Create the send button
send_button = tk.Button(
    root,
    text="Enter",
    command=send_message,
    bg="light green",
    fg="black",
    font=("Arial", 10),
)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Configure the grid
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

root.mainloop()


# Start the main loop
root.mainloop()
