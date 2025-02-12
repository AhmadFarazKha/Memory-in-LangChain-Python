from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationEntityMemory,
    ConversationSummaryBufferMemory
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_API_KEY')

def create_basic_chat():
    """Creates a basic chat with conversation buffer memory"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    memory = ConversationBufferMemory()
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def create_window_chat(k=2):
    """Creates a chat with window memory (limited context)"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    memory = ConversationBufferWindowMemory(k=k)
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def create_entity_chat():
    """Creates a chat with entity memory"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    memory = ConversationEntityMemory(llm=llm)
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def create_summary_chat():
    """Creates a chat with summary buffer memory"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def create_vector_chat():
    """Creates a chat with vector store memory"""
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.7
    )
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    vectorstore = Chroma(
        collection_name="memory_store",
        embedding_function=embeddings
    )
    
    memory = VectorStoreRetrieverMemory(
        retriever=vectorstore.as_retriever(),
        memory_key="history"
    )
    
    return ConversationChain(
        llm=llm,
        memory=memory,
        verbose=True
    )

def main():
    print("Welcome to LangChain Memory Demo!")
    print("\nSelect a memory type to test:")
    print("1. Basic Memory (remembers entire conversation)")
    print("2. Window Memory (remembers last N messages)")
    print("3. Entity Memory (tracks specific entities)")
    print("4. Summary Memory (maintains a summary)")
    print("5. Vector Store Memory (semantic search)")
    
    choice = input("\nEnter your choice (1-5): ")
    
    if choice == '1':
        chat = create_basic_chat()
        print("\nBasic Memory Chat - This remembers everything you say")
        print("Type 'memory' to see what's stored, or 'quit' to exit")
        
    elif choice == '2':
        k = int(input("How many messages to remember? "))
        chat = create_window_chat(k)
        print(f"\nWindow Memory Chat - This remembers last {k} messages")
        print("Type 'memory' to see what's stored, or 'quit' to exit")
        
    elif choice == '3':
        chat = create_entity_chat()
        print("\nEntity Memory Chat - This tracks entities (people, places, things)")
        print("Try mentioning names and properties!")
        print("Type 'memory' to see entities, or 'quit' to exit")
        
    elif choice == '4':
        chat = create_summary_chat()
        print("\nSummary Memory Chat - This maintains a summary of the conversation")
        print("Type 'memory' to see the summary, or 'quit' to exit")
        
    elif choice == '5':
        chat = create_vector_chat()
        print("\nVector Store Memory Chat - This uses semantic search")
        print("Type 'memory' to see stored vectors, or 'quit' to exit")
        
    else:
        print("Invalid choice!")
        return

    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            break
            
        if user_input.lower() == 'memory':
            if isinstance(chat.memory, ConversationEntityMemory):
                print("\nStored Entities:", chat.memory.entity_store.store)
            else:
                print("\nMemory Contents:", chat.memory.load_memory_variables({})["history"])
            continue
            
        response = chat.predict(input=user_input)
        print("\nBot:", response)

if __name__ == "__main__":
    main()