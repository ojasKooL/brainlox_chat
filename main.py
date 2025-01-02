from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Pinecone
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")
pc = Pinecone(api_key=pinecone_api_key)
text_field = "text"
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
index = pc.Index(host=pinecone_host)
vectorstore = PineconeVectorStore(index=index, embedding=embed)
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

retriever = vectorstore.as_retriever()
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    You are an expert course advisor for Brainlox's technical courses (https://brainlox.com/courses/category/technical).
    Your role is to provide clear, engaging, and user-friendly course recommendations based on the information available on the site, which you have access to through the knowledge base.

    **Guidelines for Starting Responses:**
    1. Use varied and engaging openings to avoid repetitive phrases like "It seems like...".
       - Examples: "Sure! Here are some great courses you might enjoy." or "Excited to help! Based on your query, here’s what I recommend."
       - Adjust tone based on context (e.g., enthusiastic, friendly, or professional).
    2. Assume that the context provided through the retrieval system (RAG) is your inherent knowledge. Treat it as such, and provide course recommendations or answers to user queries using that information.

    **Response Guidelines:**
    - Structure recommendations in a clear format (e.g., bullet points or numbered lists).
    - Tailor responses dynamically to user preferences, skill levels, and goals based on the knowledge base.
    - Avoid generic recommendations by leveraging the full context of the knowledge base without implying the user provided any data.
    - Use an approachable tone while keeping the response concise and relevant.
    - Avoid overuse of explicit examples that could make the chatbot biased towards a specific example; aim for generality and flexibility in recommendations.
    """),
    ("human", "{context}")
])

combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

app = Flask(__name__)
CORS(app)
@app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get("query", "")
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Invoke RAG chain
        response = rag_chain.invoke({"input": query})
        
        # Serialize 'Document' objects in the context (if present)
        context = response.get("context", [])
        serialized_context = [
            {
                "id": doc.id,
                "metadata": doc.metadata,
                "page_content": doc.page_content
            } for doc in context
        ]
        
        # Return serialized response
        return jsonify({
            "query": query,
            "response": {
                "answer": response.get("answer", ""),
                "context": serialized_context
            }
        }), 200
    except Exception as e:
        print(f"Error in /chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
