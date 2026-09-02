import streamlit as st
from transformers import pipeline

# Page configuration
st.set_page_config(
    page_title="Text Generation App",
    page_icon="🤖"
)

st.title("🤖 Text Generation using AI")
st.write("Enter a prompt and let the AI generate text.")

# Load the text generation model
@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model="Qwen/Qwen3.8-2.4T-A95B"
    )
    return generator

generator = load_model()

# User input
prompt = st.text_area(
    "Enter your prompt:",
    placeholder="Example: Artificial Intelligence is"
)

# Generation settings
max_length = st.slider(
    "Maximum length",
    min_value=30,
    max_value=200,
    value=100
)

temperature = st.slider(
    "Temperature",
    min_value=0.1,
    max_value=1.5,
    value=0.7
)

# Generate button
if st.button("Generate Text 🚀"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")

    else:
        with st.spinner("Generating text..."):

            result = generator(
                prompt,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=1
            )

            generated_text = result[0]["generated_text"]

        st.subheader("Generated Text")
        st.write(generated_text)