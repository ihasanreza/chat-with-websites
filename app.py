import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

def create_vector_store_from_url(url):
    """
    Create a vector store from the text content obtained from the provided URL.

    Args:
    url (str): The URL from which to fetch the text content.

    Returns:
    vector_store: A vector store created from the text content.
    """
    # web loader to retrieve the text content from the URL
    web_loader = WebBaseLoader(url)
    text = web_loader.load()

    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(text)

    # Vector store from chunks, using OpenAI embeddings
    vector_store = Chroma.from_documents(chunks, OpenAIEmbeddings())

    return vector_store


def create_context_retriever_chain(vector_store):
    """
    Create a retriever chain for contextual information retrieval based on the provided vector store.

    Args:
    vector_store (Chroma): The vector store containing embeddings of text data.

    Returns:
    retriever_chain: A chain for contextual information retrieval.
    """
    
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()

    prompt_template = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt_template)

    return retriever_chain


def create_conversational_rag_chain(retriever_chain): 
    """
    Create a chain for conversational response generation based on the provided retriever chain.

    Args:
    retriever_chain: A chain for contextual information retrieval.

    Returns:
    rag_chain: A chain for generating responses.
    """

    llm = ChatOpenAI()

    # Prompt template for generating responses based on conversation context
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
    return rag_chain


def generate_response(user_input):
    """
    Generate a response based on the user input by leveraging contextual information from the conversation history.

    Args:
    user_input (str): The input query provided by the user.

    Returns:
    str: The response generated based on the provided user input.
    """

    retriever_chain = create_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = create_conversational_rag_chain(retriever_chain)
    
    # Invoke the RAG chain to generate a response based on the conversation history and user input
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response['answer']

# App config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")

# Sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url == "":
    st.info("Please enter a website URL")

else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = create_vector_store_from_url(website_url)

    user_query = st.chat_input("Type your message here...")

    if user_query is not None and user_query != "":
        response = generate_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # Chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
