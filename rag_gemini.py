import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder

from langchain.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

FAISS_PATH = r"faiss_index"

SYSTEM_PROMPT = """You are a helpful AI bot.
While answering, you don't use your internal knowledge,
but solely the information in the "The knowledge" section.
You don't mention anything to the user about the provided knowledge."""

CHAT_HISTORY = []

embeddings_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=api_key  
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    google_api_key=api_key
)

vector_store = FAISS.load_local(FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)

retriever = vector_store.as_retriever(search_kwargs={'k': 5})

template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        # Use a SystemMessage for the knowledge to avoid confusing the AI
        ("system", "The knowledge: {knowledge}"),
        # Use MessagesPlaceholder for a list of messages (chat history)
        MessagesPlaceholder(variable_name="messages"),
        # User's query
        ("human", "{input}"),
    ])


def rag_query(query: str):
    
    global CHAT_HISTORY
    
    docs = retriever.invoke(query)
    
    if docs:
        knowledge = "\n".join([doc.page_content for doc in docs])

    
    prompt_value = template.invoke(
        {
            "messages": CHAT_HISTORY,
            "knowledge": knowledge if docs else "No relevant knowledge found.",
            "input": query
        }
    )

    print("--- PROMPT SENT TO LLM ---")
    print(prompt_value.to_string())
    print("--------------------------")

    response = llm.invoke(prompt_value)
    print(response)
    
    CHAT_HISTORY.append(HumanMessage(content=query))
    CHAT_HISTORY.append(AIMessage(content=response.content))
    
    return response.content

while True:
    query = input("Enter your query (or 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    response = rag_query(query)
    print(f"Response: {response}")