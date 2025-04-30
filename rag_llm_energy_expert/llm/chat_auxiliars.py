from google import genai
from google.genai import types
from loguru import logger


import sys

sys.path.append("../..")

from rag_llm_energy_expert.credentials import get_llm_config
from rag_llm_energy_expert.config import GCPConfig, QdrantConfig
from rag_llm_energy_expert.search.searchers import semantic_search


# Initialize the config classes
qdrant_config = QdrantConfig()
llm_config = get_llm_config()
gcp_config = GCPConfig()

# Initialize the GenAi client
genai_client = genai.Client(
    api_key=llm_config.API_KEY.get_secret_value(),
)

# Create the system prompt that all the chat sessions will have
main_system_prompt = (
    "You are an energy expert. You are a helpful assistant that provides information about energy-related topics."
    "If you don't know the answer, say that you do not have the information available, and that if the user thinks the info"
    "should be embedded into this chatbot, contact Emmanuel Amador, which email is emmanuel_mald@hotmail.com, who is also the owner of"
    "this chatbot. Answer all the questions in the same language that the user asks, and use the context provided on each query."
    "Try to create answers that are less than 10 lines long."
)


def create_chat_session(
    history: list,
    model: str = llm_config.MODEL,
    temperature: float = llm_config.TEMPERATURE,
    system_prompt: str = main_system_prompt,
) -> genai.chats.Chat:
    """
    Create a new chat session with the GenAI API.

    Args:
        temperature (float): The temperature for the LLM response generation.
            Higher values make the output more random, while lower values make it more deterministic.
        history (list): The history of the chat session. This is used to maintain context.
                EX:  [
    {"role": "user", "parts": [{"text": "When did the Mexican energy reform start?"}]},
    {"role": "model", "parts": [{"text": "The Mexican energy reform started in 2013."}]},

    Returns:
        genai.chats.Chat: The chat session object.
    """
    logger.info("Creating a new chat session...")

    # Create a new chat session
    chat_session = genai_client.chats.create(
        model=model,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt,
        ),
        history=history,
    )
    logger.info("Chat session created successfully.")
    return chat_session


def generate_response(
    prompt: str,
    chat_session: genai.chats.Chat,
    temperature: float = llm_config.TEMPERATURE,
    chunk_overlap: int = qdrant_config.CHUNK_OVERLAP,
    embedding_model_name: str = qdrant_config.EMBEDDING_MODEL_NAME,
    collection_name: str = qdrant_config.COLLECTION_NAME
    + qdrant_config.COLLECTION_VERSION,
    documents_limit: int = qdrant_config.DOCUMENTS_RETRIEVED_LIMIT,
    system_prompt: str = main_system_prompt,
) -> str:
    """
    Generate a response from the LLM using the GenAI API.

    Args:
        prompt (str): The input prompt for the LLM.
        chat_session (genai.chats.Chat): The chat session object for maintaining context.
        temperature (float): The temperature for the LLM response generation.
            Higher values make the output more random, while lower values make it more deterministic.
        chunk_overlap (int): The overlap between chunks of text for semantic search.
        embedding_model_name (str): The name of the embedding model to use for semantic search.
        collection_name (str): The name of the Qdrant collection to search in.
        documents_limit (int): The maximum number of documents to retrieve from Qdrant.

    Returns:
        str: The generated response from the LLM.
    """
    # Based on the user's prompt, search for the context in the Qdrant database
    # and get the most relevant context to provide to the LLM.
    logger.info("Generating response...")
    logger.info("Retrieving context...")
    context = semantic_search(
        query=prompt,
        documents_limit=documents_limit,
        collection_name=collection_name,
        chunk_overlap=chunk_overlap,
        embedding_model_name=embedding_model_name,
    )
    logger.info("Context retrieved successfully.")
    chat_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=f"{system_prompt}\n\nContext: {context}",
    )

    logger.info("Generating response...")
    response = chat_session.send_message(message=prompt, config=chat_config)
    logger.info("Response generated successfully.")

    return response.text
