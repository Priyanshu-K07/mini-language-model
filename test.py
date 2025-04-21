import streamlit as st
import torch
from lm import LanguageModel, encode 

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = LanguageModel()
model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

context = "\n"

st.title("Character Level Language Model")
st.subheader("Trained on Shakespeare's work")
max_new_tokens = st.text_input(label="", placeholder="Enter number of characters to generate (default: 100)")

# When user hits the button
if st.button("Generate"):
    max_new_tokens = 100 if max_new_tokens=="" else int(max_new_tokens)
    
    # Encode prompt
    idx = torch.tensor([encode(context)], dtype=torch.long).to(device)
    
    # Output placeholder
    output = st.empty()
    generated_text = context

    # Use generator to stream output
    for token in model.generate(idx, max_new_tokens=max_new_tokens):  
        generated_text += token
        output.text(generated_text)
        output.markdown(f"```{generated_text}▍")

    output.markdown(f"```{generated_text}")
