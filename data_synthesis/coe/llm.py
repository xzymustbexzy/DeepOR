import os
import time
from functools import lru_cache

from openai import AzureOpenAI, OpenAI


# Default config. Override with environment variables if needed.
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_KEY = ""
DEFAULT_AZURE_API_VERSION = "2025-04-01-preview"
DEFAULT_AZURE_ENDPOINT = ""
DEFAULT_BASE_URL = "https://api.deepseek.com"


def _read_env(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def get_llm_config():
    return {
        "provider": _read_env("LLM_PROVIDER", "openai").lower(),
        "model": _read_env("LLM_MODEL", DEFAULT_MODEL),
        "api_key": _read_env("LLM_API_KEY", DEFAULT_API_KEY),
        "azure_api_version": _read_env("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION),
        "azure_endpoint": _read_env("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT),
        "base_url": _read_env("LLM_BASE_URL", DEFAULT_BASE_URL),
    }


def get_default_model():
    return get_llm_config()["model"]


@lru_cache(maxsize=1)
def get_llm_client():
    config = get_llm_config()
    provider = config["provider"]

    if provider == "azure":
        return AzureOpenAI(
            api_key=config["api_key"],
            api_version=config["azure_api_version"],
            azure_endpoint=config["azure_endpoint"],
        )

    base_url = config.get("base_url")
    if base_url:
        return OpenAI(api_key=config["api_key"], base_url=base_url)
    return OpenAI(api_key=config["api_key"])


def create_chat_completion(messages, model=None, max_retries=3, retry_delay=100, **kwargs):
    client = get_llm_client()
    target_model = model or get_default_model()

    kwargs.setdefault("temperature", 0)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                **kwargs,
            )
            return response
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print(f"LLM call failed after max retries ({max_retries})")
                raise
