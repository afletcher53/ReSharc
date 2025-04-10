# src/llm_utils.py

import os
import time
import sys
from dotenv import load_dotenv

# --- Google GenAI Imports ---
import google.generativeai as genai
from google.generativeai.types import generation_types, safety_types

# --- OpenRouter Imports ---
# Use 'openai' library configured for OpenRouter
# pip install openai
try:
    from openai import (
        OpenAI,
        APIError,
        RateLimitError,
        APIConnectionError,
    )  # Import specific errors
except ImportError:
    print(
        "Warning: 'openai' library not installed. OpenRouter functionality will not be available."
    )
    print("Install it using: pip install openai")
    OpenAI = None  # Define as None if import fails

# Load environment variables
load_dotenv()

# ==================================================
# Google Generative AI Functions
# ==================================================


def configure_google_genai(settings):
    """Configures the Google Generative AI client using an API key from env."""
    api_key_env_var = settings.get("api_key_env_var", "GOOGLE_API_KEY")
    api_key = os.getenv(api_key_env_var)

    if not api_key:
        print(
            f"Error: Google API Key environment variable '{api_key_env_var}' not found."
        )
        return False  # Indicate failure

    try:
        genai.configure(api_key=api_key)
        print("Google Generative AI client configured successfully.")
        return True  # Indicate success
    except Exception as e:
        print(f"Error configuring Google Generative AI client: {e}")
        return False


def call_google_genai_llm(model_name, prompt, settings, max_retries=2, delay=5):
    """
    Calls a Google Generative AI model (Gemini) and returns content and token counts.

    Args:
        model_name (str): The name of the Google model.
        prompt (str): The input prompt.
        settings (dict): The google_settings dictionary from config.
        max_retries (int): Max retry attempts.
        delay (int): Delay between retries.

    Returns:
        tuple: (content, prompt_tokens, completion_tokens) or (None, 0, 0) on failure.
    """
    print(f"--- Calling Google GenAI model: {model_name} ---")
    attempt = 0

    gen_config_dict = settings.get("generation_config")
    safety_settings_dict = settings.get("safety_settings")
    generation_config = None
    safety_settings = None

    # --- Process Google Generation Config ---
    if gen_config_dict:
        try:
            generation_config = generation_types.GenerationConfig(**gen_config_dict)
            print(f"Using Google Generation Config: {gen_config_dict}")
        except Exception as e:
            print(f"Warning: Could not apply google generation_config: {e}")

    # --- Process Google Safety Settings ---
    if safety_settings_dict:
        try:
            processed_safety = {}
            for key, value in safety_settings_dict.items():
                try:
                    harm_category = getattr(safety_types.HarmCategory, key)
                    block_threshold = getattr(safety_types.HarmBlockThreshold, value)
                    processed_safety[harm_category] = block_threshold
                except AttributeError:
                    print(
                        f"Warning: Invalid safety setting key/value for Google: {key}={value}. Skipping."
                    )
            if processed_safety:
                safety_settings = processed_safety
                print(f"Using Google Safety Settings: {safety_settings_dict}")
        except Exception as e:
            print(f"Warning: Could not apply google safety_settings: {e}")

    # --- Initialize Google Model ---
    try:
        model = genai.GenerativeModel(
            model_name,
            generation_config=generation_config if generation_config else None,
            safety_settings=safety_settings if safety_settings else None,
        )
    except Exception as e:
        print(f"Error initializing Google GenerativeModel '{model_name}': {e}")
        return None, 0, 0

    # --- API Call Loop ---
    while attempt <= max_retries:
        try:
            prompt_token_count = model.count_tokens(prompt).total_tokens
            print(
                f"Sending request to Google model {model_name} (Attempt {attempt + 1}/{max_retries + 1})..."
            )
            response = model.generate_content(prompt)

            content = None
            completion_token_count = 0

            if not response.candidates:
                print(
                    f"Warning: Google response blocked/empty. Feedback: {response.prompt_feedback}"
                )
                raise generation_types.BlockedPromptException(
                    f"Prompt blocked: {response.prompt_feedback}"
                )

            finish_reason_obj = response.candidates[0].finish_reason
            finish_reason_name = getattr(finish_reason_obj, "name", None)

            if finish_reason_name != "STOP":
                reason_display = (
                    finish_reason_name if finish_reason_name else str(finish_reason_obj)
                )
                print(
                    f"Warning: Google generation finished unexpectedly. Reason: {reason_display}"
                )

            try:
                content = response.text
                print(f"--- Google response received (Length: {len(content)}) ---")
            except ValueError:
                print(f"Warning: Could not extract text content from Google response.")
                content = None

            if content:
                completion_token_count = model.count_tokens(
                    response.candidates[0].content
                ).total_tokens
            else:
                completion_token_count = 0

            print(
                f"--- Google Usage: Prompt Tokens≈{prompt_token_count}, Completions Tokens≈{completion_token_count} ---"
            )
            return (
                content.strip() if content else None,
                prompt_token_count,
                completion_token_count,
            )

        # --- Google Error Handling ---
        except generation_types.BlockedPromptException as bpe:
            print(f"Error: Google prompt blocked. {bpe}")
            attempt += 1
            if attempt <= max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for blocked Google prompt. Skipping.")
                return None, 0, 0
        except Exception as e:
            import traceback

            print(
                f"Error calling Google model {model_name} (Attempt {attempt + 1}/{max_retries + 1}): {e}"
            )
            # traceback.print_exc()
            attempt += 1
            if attempt <= max_retries:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"Max retries reached for Google model. Skipping.")
                return None, 0, 0
    return None, 0, 0


# ==================================================
# OpenRouter Functions (using OpenAI library)
# ==================================================


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
            timeout=60.0,  # Add a default timeout
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
                max_tokens=gen_config.get("max_tokens", 2048),
                temperature=gen_config.get("temperature", 0.7),
                top_p=gen_config.get("top_p"),  # Pass None if not set
                frequency_penalty=gen_config.get(
                    "frequency_penalty"
                ),  # Pass None if not set
                presence_penalty=gen_config.get(
                    "presence_penalty"
                ),  # Pass None if not set
                # Add other compatible parameters from gen_config if needed
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

        # --- OpenRouter Error Handling ---
        except RateLimitError as rle:
            print(f"Error: OpenRouter Rate Limit Exceeded for {model_name}. {rle}")
            # Rate limits often require longer delays or stopping
            attempt = max_retries + 1  # Stop retrying on rate limit
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
            # traceback.print_exc()

        # Common retry logic for non-rate-limit errors
        attempt += 1
        if attempt <= max_retries:
            print(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Max retries reached for OpenRouter model {model_name}. Skipping.")
            return None, 0, 0

    return None, 0, 0
