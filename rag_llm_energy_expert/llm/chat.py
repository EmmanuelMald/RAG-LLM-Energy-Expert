from google import genai
from google.genai import types
from loguru import logger
import sys

sys.path.append("../..")

from rag_llm_energy_expert.credentials import get_qdrant_config, get_llm_config
from rag_llm_energy_expert.search.searchers import semantic_search


# Initialize the config classes
qdrant_config = get_qdrant_config()
llm_config = get_llm_config()

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


def create_chat_session(temperature: float = 0.05) -> genai.chats.Chat:
    """
    Create a new chat session with the GenAI API.

    Args:
        temperature (float): The temperature for the LLM response generation.
            Higher values make the output more random, while lower values make it more deterministic.

    Returns:
        genai.chats.Chat: The chat session object.
    """
    logger.info("Creating a new chat session...")
    # Create a new chat session
    chat_session = genai_client.chats.create(
        model=llm_config.MODEL,
        config=types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=main_system_prompt,
        ),
    )
    logger.info("Chat session created successfully.")
    return chat_session


def generate_response(
    prompt: str,
    chat_session: genai.chats.Chat,
    temperature: float = 0.5,
) -> str:
    """
    Generate a response from the LLM using the GenAI API.

    Args:
        prompt (str): The input prompt for the LLM.
        chat_session (genai.chats.Chat): The chat session object for maintaining context.
        temperature (float): The temperature for the LLM response generation.
            Higher values make the output more random, while lower values make it more deterministic.

    Returns:
        str: The generated response from the LLM.
    """
    # Based on the user's prompt, search for the context in the Qdrant database
    # and get the most relevant context to provide to the LLM.
    logger.info("Generating response...")
    logger.info("Retrieving context...")
    context = semantic_search(
        query=prompt,
        documents_limit=qdrant_config.DOCUMENTS_RETRIEVED_LIMIT,
        collection_name=qdrant_config.COLLECTION_NAME
        + qdrant_config.COLLECTION_VERSION,
        chunk_overlap=0,
        embedding_model_name=None,
    )
    logger.info("Context retrieved successfully.")
    chat_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=f"{main_system_prompt}\n\nContext: {context}",
    )

    logger.info("Generating response...")
    response = chat_session.send_message(message=prompt, config=chat_config)
    logger.info("Response generated successfully.")

    return response.text
