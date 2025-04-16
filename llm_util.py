from openai import OpenAI

def get_response_from_llm(user_prompt, temperature=.3, debug_mode=False):
    llm_server_url = "http://localhost:1234/v1"
    openai_key = "lm-studio"
    model_name = "llama3.2"
    # Point OpenAI to local LLM server
    client = OpenAI(base_url=llm_server_url, api_key=openai_key)

    response = client.chat.completions.create(
        model = model_name,  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    if debug_mode == True:
        print(f"Input token count {response.usage.prompt_tokens}")
        print(f"Output token count - {response.usage.completion_tokens}")
        print(f"Total token count - {response.usage.total_tokens}")
        print(response.choices[0].message.content)
    return response

def get_text_from_llm(user_prompt, temperature=.1, debug_mode=False):
    response = get_response_from_llm(user_prompt, temperature, debug_mode)
    return response.choices[0].message.content

if __name__=="__main__":
    print(get_text_from_llm("where is SFO?"))
