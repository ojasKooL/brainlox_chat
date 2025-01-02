# brainlox_chat

Brainlox Chat is an AI chatbot built with Flask, LangChain, and Groq to provide personalized course recommendations from Brainlox’s technical courses.

## Features

- **Personalized Course Recommendations**: Dynamic suggestions based on user queries.
- **Pinecone Integration**: Uses Pinecone for vector storage and retrieval.
- **Groq LLM**: Generates responses using a powerful language model.
- **Web Interface**: Simple UI for querying the chatbot.

## Technologies

- **Flask**: Web framework
- **LangChain**: LLM and document chain management
- **Pinecone**: Vector store for course data
- **HuggingFace**: Embeddings for semantic search
- **Groq**: LLM for response generation

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/brainlox_chat.git
   cd brainlox_chat
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with the following variables:
   ```
   GROQ_API_KEY=your_groq_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_HOST=your_pinecone_host
   ```

5. Run the app:
   ```bash
   python app.py
   ```

## Usage

Send a POST request to `/chat` with a JSON payload:

```json
{
  "query": "Tell me about AI courses"
}
```

Response:

```json
{
  "query": "Tell me about AI courses",
  "response": {
    "answer": "Here are some great AI courses...",
    "context": [...]
  }
}
```
