import os
import re 
import openai

client = openai.OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

CODE_BLOCK_PATTERN = r"```(\w*)\n(.*?)\n```"
def extract_code(text: str, pattern: str = CODE_BLOCK_PATTERN):
    match = re.findall(pattern, text, flags=re.DOTALL)
    return match if match else []


def call_openai_completion(**kwargs):
    retry_count = 0
    while retry_count <= 20:
        try:
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        
        except openai.RateLimitError as error:
            print(f"OpenAI API retry for {retry_count} times ({error})")
            time.sleep(5)
            retry_count += 1
            continue

        except Exception as error:
            print(error)
            return None
        
    print('Retry limit reached.')
    return None