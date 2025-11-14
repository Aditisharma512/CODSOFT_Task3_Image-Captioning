# streamlit_app.py
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.title("🖼️ Image Caption Generator — Botty")
st.write("Upload an image and click Generate Caption")

uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    if st.button("Generate caption"):
        with st.spinner("Generating caption..."):
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)
            st.success("Caption:")
            st.write(caption)
