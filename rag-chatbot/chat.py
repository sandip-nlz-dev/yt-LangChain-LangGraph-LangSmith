from langchain_qdrant import QdrantVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

def call_llm(messages):
    model = init_chat_model("google_genai:gemini-2.5-flash-lite")
    response = model.invoke(messages)
    return response.content

# vector store 
vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="demo_rag",
    url="http://localhost:6333",
)

# Chat loop
conversation_history = []

while True:
    query = input("Enter your question (or 'exit' to end): ")
    if query.lower() == 'exit':
        break

    #retrive relavent data chunks base on user query from vector database
    relevant_chunks = vector_store.similarity_search(query=query)

    # Define the system prompt for the chatbot
    system_prompt = f"""
    You are a helpful assistant. You help the user to find the answer to their question based on the provided context.
    context: {relevant_chunks}
    You will be provided with a context and a question. You need to answer the question based on the context.
    If the context does not provide enough information to answer the question, you should say "I don't know".
    Note:
    Answer should be in detaild and should not be too short.
    Answer should be in a conversational tone.
    """

    messages = [{"role": "system", "content": system_prompt}]

    # # Add conversation history to messages
    for msg in conversation_history:
        messages.append(msg)

    # Add current user query to messages    
    messages.append({"role": "user", "content": query})

    response = call_llm(messages)

    # Update conversation history
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": response})

    # # Keep conversation history limited to last 4 interactions (8 messages)
    if len(conversation_history) > 8:
        conversation_history = conversation_history[-8:]

    print("\nAssistant:", response, "\n")
