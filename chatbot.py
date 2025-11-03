import os, sys
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import streamlit as st

INDEX_DIR = "data/index"

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
vectordb = Chroma(persist_directory=INDEX_DIR, embedding_function=None)
retriever = vectordb.as_retriever(search_kwargs={"k": 4})
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# --- CLI mode ---
if len(sys.argv) > 1 and sys.argv[1] == "cli":
    while True:
        q = input("\nAsk about Medicare plans (or 'exit'): ")
        if q.lower() in ["exit", "quit"]: break
        ans = qa_chain.invoke(q)
        print("\n", ans["result"])
    sys.exit(0)

# --- Web UI ---
st.set_page_config(page_title="Medicare Plan Chatbot", page_icon="💬", layout="wide")
st.title("💬 Medicare Plan Chatbot")
query = st.text_input("Ask about plan benefits:")

if query:
    ans = qa_chain.invoke(query)
    st.markdown("### Answer")
    st.write(ans["result"])
    st.markdown("---")

    st.markdown("**Sources:**")
    for d in ans["source_documents"]:
        src = d.metadata.get("source")
        pg  = d.metadata.get("page")
        st.text(f"{src}, page {pg}")
