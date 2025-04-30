from loguru import logger
from datetime import datetime
from google import genai
import json
import re

import sys

sys.path.append("../..")

from rag_llm_energy_expert.utils.gcp.bigquery import query_data, insert_rows, update_row
from rag_llm_energy_expert.config import GCPConfig, LLMConfig
from rag_llm_energy_expert.llm.chat_auxiliars import main_system_prompt


gcp_config = GCPConfig()
llm_config = LLMConfig()


def prepare_chat_history(chat_session: genai.chats.Chat) -> list:
    """
    Convert the chat session history into a format that can be stored in BigQuery:

    Args:
        chat_session (genai.chats.Chat): The chat session object.

    Returns:
        list: A list of dictionaries containing the chat history.
    """
    logger.info("Preparing chat history for BigQuery...")

    chat_history = []
    for message in chat_session.get_history():
        role = message.role
        text = message.parts[0].text
        chat_history.append({"role": role, "parts": [{"text": text}]})

    # Convert the chat history to a JSON string
    chat_history = json.dumps(chat_history, ensure_ascii=False)
    logger.info("Chat history prepared to be ingested in BigQuery.")

    return chat_history


def insert_user_data(
    full_name: str,
    company_name: str,
    email: str,
    company_role: str,
    dataset_id=gcp_config.BQ_DATASET,
    table_id=gcp_config.BQ_USERS_TABLE,
    project_id=gcp_config.PROJECT_ID,
    table_pk=gcp_config.BQ_USERS_PK,
) -> str:
    """
    Insert user data into the BigQuery database.

    Args:
        full_name (str): The full name of the user.
        company_name (str): The name of the user's company.
        email (str): The email address of the user.
        company_role (str): The role of the user in the company.
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the BigQuery table.
        project_id (str): The ID of the GCP project.
        table_pk (str): The primary key of the BigQuery table.

    Returns:
        str -> user_id that was inserted into the BigQuery table.
    """
    logger.info("Inserting user data into BigQuery...")

    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime(r"%Y-%m-%d %H:%M:%S")

    # Cleaning the data
    full_name = full_name.strip().capitalize()
    email = email.strip()
    company_name = company_name.strip().capitalize()
    company_role = company_role.strip().lower()

    logger.info("Checking if the user already exists...")
    # If the user does not exist, the row_iterator will be an empty list
    query_is_user = f"""
            select
                {table_pk}
                
            from {project_id}.{dataset_id}.{table_id}
            where full_name = '{full_name}' and email = '{email}'
        """
    # Query the BigQuery database and return an iterator of rows
    is_user_iterator = query_data(query_is_user)

    # Generate a list of users, in this case, it can be either an empty list or a list with one element
    list_user_id = [row[table_pk] for row in is_user_iterator]

    if len(list_user_id) > 0:
        logger.info("User already exists in the database.")
        update_row(
            table_name=table_id,
            dataset_name=dataset_id,
            project_id=project_id,
            primary_key_column_name=table_pk,
            row_id=list_user_id[0],
            update_data={
                "last_entered_at": current_time,
            },
        )
        return list_user_id[0]

    logger.info("Generating a new user ID...")
    query_count_users = f"""
            select
                count(*) as total_users
            from {project_id}.{dataset_id}.{table_id}
    """

    # Query the BigQuery database to get the total number of users
    rows = query_data(query=query_count_users)
    total_users = [row.total_users for row in rows][0]

    # Generating the user ID
    next_id = total_users + 1
    user_id = f"UID{next_id:05d}"
    logger.info(f"Generated user ID: {user_id}")

    # Preparing the columns to fill in the BigQuery table
    data_to_insert = {
        "user_id": user_id,
        "full_name": full_name,
        "company_name": company_name,
        "email": email,
        "company_role": company_role,
        "created_at": current_time,
        "last_entered_at": current_time,
    }

    # Insert the data into the BigQuery table
    insert_rows(
        project_id=project_id,
        dataset_name=dataset_id,
        table_name=table_id,
        rows=[
            data_to_insert,
        ],
    )

    return user_id


def insert_llms_data(
    last_user_id: str,
    system_prompt: str = main_system_prompt,
    llm_model_name: str = llm_config.MODEL,
    temperature: float = llm_config.TEMPERATURE,
    project_id: str = gcp_config.PROJECT_ID,
    dataset_id: str = gcp_config.BQ_DATASET,
    table_id: str = gcp_config.BQ_LLMS_TABLE,
    table_pk: str = gcp_config.BQ_LLMS_PK,
) -> str:
    """
    Insert LLM data into the BigQuery database.

    Args:
        last_user_id (str): The ID of the user who last used the LLM.
        system_prompt (str): The system prompt for the LLM.
        llm_model_name (str): The name of the LLM model.
        temperature (float): The temperature for the LLM response generation.
        project_id (str): The ID of the GCP project.
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the BigQuery table.
        table_pk (str): The primary key of the BigQuery table.

    Returns:
        str: The ID of the LLM version that was inserted into the BigQuery table.
    """
    logger.info("Inserting LLM data into BigQuery...")

    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime(r"%Y-%m-%d %H:%M:%S")

    # Cleaining the data
    system_prompt = system_prompt.strip().lower()
    llm_model_name = llm_model_name.strip()
    temperature = round(temperature, 4)

    # Checking if the LLM model already exists
    logger.info("Checking if the LLM model already exists...")

    # Use "" instead of '' for the system prompt to avoid issues with single quotes in the text
    query_is_llm = f"""
            select
                {table_pk}
                
            from {project_id}.{dataset_id}.{table_id}
            where llm_model_name = '{llm_model_name}' and temperature = {temperature} and system_prompt = "{system_prompt}" 
        """

    rows = query_data(query_is_llm)

    llm_version_id = [row.llm_version_id for row in rows]

    if len(llm_version_id) > 0:
        logger.info("LLM model already exists in the database.")
        update_row(
            table_name=table_id,
            dataset_name=dataset_id,
            project_id=project_id,
            primary_key_column_name=table_pk,
            row_id=llm_version_id[0],
            update_data={
                "last_used_at": current_time,
                "last_user_id": last_user_id,
            },
        )
        return llm_version_id[0]

    logger.info("Generating a new llm_version_id...")
    query_count_users = f"""
            select
                count(*) as model_versions
            from {project_id}.{dataset_id}.{table_id}
            where llm_model_name = '{llm_model_name}'
    """

    # Query the BigQuery database to get the total number of users
    rows = query_data(query=query_count_users)
    total_model_versions = [row.model_versions for row in rows][0]

    # Generating the llm version ID
    next_id = total_model_versions + 1
    llm_version_id = f"{llm_model_name}-v{next_id:05d}"
    logger.info(f"Generated llm version ID: {llm_version_id}")

    # Preparing the columns to fill in the BigQuery table
    data_to_insert = {
        "llm_version_id": llm_version_id,
        "llm_model_name": llm_model_name,
        "temperature": temperature,
        "system_prompt": system_prompt,
        "last_user_id": last_user_id,
        "created_at": current_time,
        "last_used_at": current_time,
    }

    # Insert the data into the BigQuery table
    insert_rows(
        project_id=project_id,
        dataset_name=dataset_id,
        table_name=table_id,
        rows=[
            data_to_insert,
        ],
    )

    return llm_version_id


def insert_chat_session_data(
    llm_version_id: str,
    user_id: str,
    dataset_id: str = gcp_config.BQ_DATASET,
    table_id: str = gcp_config.BQ_CHAT_SESSIONS_TABLE,
    project_id: str = gcp_config.PROJECT_ID,
    table_pk: str = gcp_config.BQ_CHAT_SESSIONS_PK,
) -> str:
    """
    Insert chat session data into the BigQuery database. This function will always generate a new chat session.

    Args:
        llm_version_id (str): The ID of the LLM version used in the chat session.
        user_id (str): The ID of the user who initiated the chat session.
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the BigQuery table.
        project_id (str): The ID of the GCP project.
        table_pk (str): The primary key of the BigQuery table.

    Returns:
        str: The ID of the chat session that was inserted into the BigQuery table.
    """
    logger.info("Inserting chat session data into BigQuery...")

    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime(r"%Y-%m-%d %H:%M:%S")

    # Cleaning the data
    llm_version_id = llm_version_id.strip()
    user_id = user_id.strip()

    logger.info("Generating a new chat session ID...")
    query_users_chats = f"""
            select
                count(*) as total_chat_sessions
            from {project_id}.{dataset_id}.{table_id}
            where user_id = '{user_id}' 
    """

    # Query the BigQuery database to get the total number of users
    rows = query_data(query=query_users_chats)
    total_chat_sessions = [row.total_chat_sessions for row in rows][0]

    # Generating the chat session ID
    next_id = total_chat_sessions + 1

    # Extracting the user number from the user_id to generate a session_id
    match = re.search(r"\d+", user_id)
    user_number = int(match.group(0))

    chat_session_id = f"CSID{user_number}-{next_id:03d}"
    logger.info(f"Generated chat session ID: {chat_session_id}")

    # Preparing the columns to fill in the BigQuery table
    data_to_insert = {
        "session_id": chat_session_id,
        "llm_version_id": llm_version_id,
        "user_id": user_id,
        "session_history": json.dumps([]),  # Because is a new chat session
        "created_at": current_time,
        "last_used_at": current_time,
    }

    insert_rows(
        project_id=project_id,
        dataset_name=dataset_id,
        table_name=table_id,
        rows=[
            data_to_insert,
        ],
    )

    return chat_session_id


def insert_prompt_data(
    session_id: str,
    prompt: str,
    llm_response: str,
    temperature: float = llm_config.TEMPERATURE,
    dataset_id: str = gcp_config.BQ_DATASET,
    table_id: str = gcp_config.BQ_PROMPTS_TABLE,
    project_id: str = gcp_config.PROJECT_ID,
) -> str:
    """
    Insert prompt data into the BigQuery database.

    Args:
        session_id (str): The ID of the chat session.
        prompt (str): The prompt text introduced by the user.
        llm_response (str): The response generated by the LLM.
        temperature (float): The temperature for the LLM response generation.
        dataset_id (str): The ID of the BigQuery dataset.
        table_id (str): The name of the BigQuery table.
        project_id (str): The ID of the GCP project.
        table_pk (str): The primary key of the BigQuery table.

    Returns:
        str -> The ID of the prompt that was inserted into the BigQuery table.
    """
    logger.info("Inserting prompt data into BigQuery...")

    # Get the current date and time
    now = datetime.now()
    current_time = now.strftime(r"%Y-%m-%d %H:%M:%S")

    logger.info("Getting the total number of prompts")
    # Get the number of prompts that exists in the table
    query_count_prompts = f"""
            select
                count(*) as total_prompts
            from {project_id}.{dataset_id}.{table_id}
    """

    # Query the BigQuery database to get the total number of prompts
    rows = query_data(query=query_count_prompts)

    logger.info("Creating a prompt_id")
    total_prompts = [row.total_prompts for row in rows][0]

    prompt_id = f"PID{total_prompts + 1:05d}"

    logger.info(f"Prompt ID: {prompt_id}")

    # Preparing the columns to fill in the BigQuery table
    data_to_insert = {
        "prompt_id": prompt_id,
        "session_id": session_id,
        "prompt": prompt,
        "llm_response": llm_response,
        "prompt_created_at": current_time,
        "llm_temperature": temperature,
    }

    # Insert the data into the BigQuery table
    insert_rows(
        project_id=gcp_config.PROJECT_ID,
        dataset_name=gcp_config.BQ_DATASET,
        table_name=gcp_config.BQ_PROMPTS_TABLE,
        rows=[
            data_to_insert,
        ],
    )

    return prompt_id
