import os
import time

from dotenv import load_dotenv
from openai import APIStatusError


try:
    from openai import (
        OpenAI,
        APIError,
        RateLimitError,
        APIConnectionError,
    )
except ImportError:
    print(
        "Warning: 'openai' library not installed. OpenRouter functionality will not be available."
    )
    print("Install it using: pip install openai")
    OpenAI = None


load_dotenv()


def setup_openrouter_client(settings):
    """Sets up the OpenAI client configured for OpenRouter."""
    if OpenAI is None:
        print(
            "Error: 'openai' library is not installed. Cannot setup OpenRouter client."
        )
        return None

    api_key_env_var = settings.get("api_key_env_var", "OPENROUTER_API_KEY")

    api_key = os.getenv(api_key_env_var)
    api_key = (
        "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"
    )
    if not api_key:
        print(
            f"Error: OpenRouter API Key environment variable '{api_key_env_var}' not found."
        )
        return None

    site_url = settings.get("site_url", "http://localhost")
    app_name = settings.get("app_name", "arc-grpo-trainer")

    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
            default_headers={
                "HTTP-Referer": site_url,
                "X-Title": app_name,
            },
            timeout=60.0,
        )
        print("OpenRouter client configured successfully.")
        return client
    except Exception as e:
        print(f"Error configuring OpenRouter client: {e}")
        return None


def call_openrouter_llm(client, model_name, prompt, settings, max_retries=2, delay=5):
    """
    Calls an OpenRouter model using the configured client.

    Args:
        client: The initialized OpenAI client for OpenRouter.
        model_name (str): The OpenRouter model identifier.
        prompt (str): The input prompt.
        settings (dict): The openrouter_settings dictionary from config.
        max_retries (int): Max retry attempts.
        delay (int): Delay between retries.

    Returns:
        tuple: (content, prompt_tokens, completion_tokens) or (None, 0, 0) on failure.
    """
    if client is None:
        print("Error: OpenRouter client is not initialized.")
        return None, 0, 0

    print(f"--- Calling OpenRouter model: {model_name} ---")
    attempt = 0
    gen_config = settings.get("generation_config", {})
    print(f"Using OpenRouter Generation Config: {gen_config}")

    while attempt <= max_retries:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=gen_config.get("max_tokens", 32000),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p"),
                frequency_penalty=gen_config.get("frequency_penalty"),
                presence_penalty=gen_config.get("presence_penalty"),
            )

            content = None
            prompt_tokens = 0
            completion_tokens = 0

            if response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                print(
                    f"--- OpenRouter Usage: Prompt Tokens={prompt_tokens}, Completion Tokens={completion_tokens} ---"
                )

            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                print(
                    f"--- OpenRouter response received (Length: {len(content if content else '')}) ---"
                )
            else:
                print(
                    f"Warning: No valid response choices received from OpenRouter model {model_name}."
                )

            return (
                content.strip() if content else None,
                prompt_tokens,
                completion_tokens,
            )

        except RateLimitError as rle:
            print(f"Error: OpenRouter Rate Limit Exceeded for {model_name}. {rle}")

            attempt = max_retries + 1
            print("Stopping retries due to rate limit.")
            return None, 0, 0
        except APIError as apie:
            print(
                f"Error: OpenRouter API Error for {model_name} (Attempt {attempt + 1}/{max_retries + 1}): {apie}"
            )
        except APIConnectionError as ace:
            print(
                f"Error: OpenRouter Connection Error for {model_name} (Attempt {attempt + 1}/{max_retries + 1}): {ace}"
            )
        except Exception as e:
            import traceback

            print(
                f"Error calling OpenRouter model {model_name} (Attempt {attempt + 1}/{max_retries + 1}): {e}"
            )

        attempt += 1
        if attempt <= max_retries:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Max retries reached for OpenRouter model {model_name}. Skipping.")
            return None, 0, 0

    return None, 0, 0


def setup_openai_client(api_settings):
    """Sets up the OpenAI client using an API key from environment variables."""
    api_key_env_var = api_settings.get("api_key_env_var", "OPENAI_API_KEY")
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        print(
            f"Error: Environment variable {api_key_env_var} for OpenAI API key not set."
        )
        return None
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        print("OpenAI client configured successfully.")
        return client
    except APIConnectionError as e:
        print(f"OpenAI APIConnectionError: Failed to connect to OpenAI API: {e}")
    except RateLimitError as e:
        print(f"OpenAI RateLimitError: OpenAI API request exceeded rate limit: {e}")
    except APIStatusError as e:
        print(f"OpenAI APIStatusError: OpenAI API returned an API Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during OpenAI client setup: {e}")
    return None
