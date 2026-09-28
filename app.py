import streamlit as st
from src.query import ask
from src.ingest import ingest

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Travel Insurance AI", page_icon="🛡️", layout="wide")

st.title("🛡️ Travel Insurance Policy Assistant")
st.markdown("Ask questions about insurance policies and get answers based on actual PDF documents.")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Settings")
    
    # 1. Insurer Selection
    insurer_list = ["ALLIANZ", "BUDGET-DIRECT", "COVER-MORE", "MEDIBANK","ALL"]
    selected_insurer = st.selectbox("Select Insurer", options=insurer_list)
    
    # 2. Parameter Tuning
    top_k = st.slider("Number of documents to retrieve (Top-K)", min_value=1, max_value=10, value=5)
    
    st.divider()

# --- MAIN CHAT INTERFACE ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("e.g. What is the coverage for medical expenses?"):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the RAG Query
    with st.chat_message("assistant"):
        with st.spinner("Searching policy documents..."):
            try:
                # Prepare filter: if "All" is selected, pass None to the query function
                insurer_filter = None if selected_insurer == "All" else selected_insurer
                
                # CALL THE LOGIC FROM src/query.py
                result = ask(
                    question=prompt, 
                    db_path="./data/vector_db/", 
                    collection_name="Travel_Insurance", 
                    insurer=insurer_filter, 
                    top_k=top_k
                )

                # Display the AI Answer
                st.markdown(result.answer)

                # Display the Sources in an expandable section
                with st.expander("📄 View Source Documents"):
                    for i, src in enumerate(result.sources, 1):
                        st.markdown(f"**Source {i}:** {src['source']} (Page {src['page']})")
                        st.info(src['text'])
                        st.markdown("---")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.info("Make sure Ollama is running and the models (llama3, nomic-embed-text) are installed.")

    # Add assistant response to chat history
    # We check if 'result' exists to avoid errors if the query failed
    if 'result' in locals():
        st.session_state.messages.append({"role": "assistant", "content": result.answer})
    else:
        st.session_state.messages.append({"role": "assistant", "content": "Sorry, I encountered an error."})
