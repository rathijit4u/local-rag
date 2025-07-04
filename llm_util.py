from openai import OpenAI, APIConnectionError
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
import logging
from llm_connection_error import LLMConnectionError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s [%(levelname)s][%(name)s]: %(message)s')

def get_response_from_llm(user_prompt, temperature=.3, debug_mode=False):
    llm_server_url = "http://localhost:1234/v1"
    openai_key = "lm-studio"
    model_name = "llama3.2"
    # Point OpenAI to local LLM server
    client = OpenAI(base_url=llm_server_url, api_key=openai_key)
    messages: list[ChatCompletionMessageParam] = [
        ChatCompletionAssistantMessageParam(role="assistant"
                , content="You are a helpful assistant. Use only the context provided below to answer the question. Do not use any external knowledge or assumptions."),
        ChatCompletionUserMessageParam(role="user", content=user_prompt)
    ]

    try:
        response = client.chat.completions.create(
            model = model_name,
            messages = messages,
            temperature=temperature
        )
        if debug_mode:
            logger.debug(f"Input token count {response.usage.prompt_tokens}")
            logger.debug(f"Output token count - {response.usage.completion_tokens}")
            logger.debug(f"Total token count - {response.usage.total_tokens}")
            logger.debug(f"LLM content - {response.choices[0].message.content}")
        return response
    except APIConnectionError as e:
        logger.debug(e)
        raise LLMConnectionError("Failed to connect to LLM server.", url=llm_server_url)
    except Exception as e:
        logger.debug(f"An unexpected error occurred: {e}")
        raise

def get_text_from_llm(user_prompt, temperature=.1, debug_mode=False):
    response = get_response_from_llm(user_prompt, temperature, debug_mode)
    return response.choices[0].message.content

if __name__=="__main__":
    print(get_text_from_llm("where is SFO?"))
