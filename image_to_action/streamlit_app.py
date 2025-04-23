import streamlit as st
from streamlit.components.v1 import html
from ocr.extractor import extract_text_from_image
from parser.instruction_parser import parse_instruction
from executor.action_handler import execute_action
import tempfile
import os

st.set_page_config(page_title="Image-to-Action AI", layout="centered")
st.title("🧠 Image → Instruction → Action")

uploaded_file = st.file_uploader("📤 Upload an instruction image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.image(tmp_path, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Process Instruction"):
        with st.spinner("Extracting text..."):
            text = extract_text_from_image(tmp_path)

        st.subheader("📝 Extracted Text")
        st.code(text)

        commands = parse_instruction(text)
        st.subheader("🧠 Parsed Instructions")
        st.json(commands)

        # Inside your button click block:
        st.subheader("🚀 Executing Actions")

        for cmd in commands:
            st.write(f"✅ Executing: `{cmd['action']}`")
            result = execute_action(cmd)

            if result["type"] == "iframe_group":
                tabs = st.tabs([tab["name"] for tab in result["tabs"]])
                for i, tab in enumerate(tabs):
                    with tab:
                        html(f'<iframe src="{result["tabs"][i]["url"]}" width="100%" height="600"></iframe>', height=620)
            
            elif result["type"] == "search_results":
                st.subheader(f"🔍 Results for: `{result['query']}`")
                for item in result["results"]:
                    st.markdown(f"**{item['title']}**")
                    st.write(item["text"])
                    st.markdown(f"[🔗 Open Link]({item['link']})")
                    st.markdown("---")

            elif result["type"] == "error":
                st.warning(result["message"])

            elif result["type"] == "info":
                st.success(result["message"])

            elif result["type"] == "note":
                st.info(f"📝 {result['text']}")

        # Clean up
        os.remove(tmp_path)
