import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Knowledge Assistant")
st.markdown("Upload documents and ask questions based on their content!")

# Sidebar for document upload
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or CSV files", 
        type=['pdf', 'txt', 'csv'], 
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Uploading and processing documents (Embedding)..."):
                files_payload = [
                    ("files", (file.name, file.getvalue(), file.type))
                    for file in uploaded_files
                ]
                try:
                    response = requests.post(f"{API_URL}/upload-documents", files=files_payload)
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Successfully processed {len(data['files'])} files!")
                        st.info(f"Generated and stored {data['chunks_added']} vector chunks.")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
        else:
            st.warning("Please upload at least one document.")
            
    st.markdown("---")
    st.header("System Status")
    try:
        health = requests.get(f"{API_URL}/health")
        if health.status_code == 200:
            st.success("Backend API is running: " + health.json()['status'])
        else:
            st.error("Backend API returned an error.")
    except:
        st.error("Backend API is unreachable.")

# Main area for Query
st.header("2. Ask Questions")
user_query = st.text_input("Enter your question about the uploaded documents:")

if st.button("Ask Assistant"):
    if user_query.strip():
        with st.spinner("Searching for answers... (Retrieval -> Reranking -> Generation)"):
            try:
                response = requests.post(
                    f"{API_URL}/query",
                    json={"query": user_query}
                )
                if response.status_code == 200:
                    data = response.json()
                    
                    st.subheader("Answer")
                    st.write(data["answer"])
                    
                    st.subheader("Supporting Details")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Confidence Score", f"{data['confidence']*100:.1f}%")
                    with col2:
                        sources = data["sources"]
                        if sources:
                            st.write("**Sources:**")
                            for s in sources:
                                st.write(f"- `{s}`")
                        else:
                            st.write("No specific sources used.")
                else:
                    st.error(f"Error calculating answer: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")
    else:
        st.warning("Please enter a question.")
