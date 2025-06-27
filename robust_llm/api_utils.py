import os
import traceback

import openai
from openai.types.chat.chat_completion import ChatCompletion
from pydantic import BaseModel

PRICING_PER_TOKEN = {
    "o3-mini": {
        "input": 1.1e-6,
        "output": 4.4e-6,
    },
    "o4-mini": {
        "input": 1.1e-6,
        "output": 4.4e-6,
    },
    "gpt-4o-mini": {
        "input": 0.15e-6,
        "output": 0.6e-6,
    },
    "gpt-4.1-mini": {
        "input": 0.4e-6,
        "output": 1.6e-6,
    },
}
# Initialize OpenAI client
client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def get_cost(completion: ChatCompletion, model_name: str) -> float:
    """Get the cost of the completion."""
    assert completion.usage is not None
    input_tokens = completion.usage.prompt_tokens
    output_tokens = completion.usage.completion_tokens
    return (
        input_tokens * PRICING_PER_TOKEN[model_name]["input"]
        + output_tokens * PRICING_PER_TOKEN[model_name]["output"]
    )


def get_response(
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    format: BaseModel,
    temperature: float | None,
) -> tuple[BaseModel | None, float] | None:
    """Use OpenAI's API to get a JSON response."""
    try:
        kwargs = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = client.beta.chat.completions.parse(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            response_format=format,  # type: ignore
            **kwargs,  # type: ignore
        )
        parsed = response.choices[0].message.parsed
        return parsed, get_cost(response, model_name)
    except Exception as e:
        print(f"Error getting response: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None
