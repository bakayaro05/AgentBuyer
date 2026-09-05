"""
Central configuration for the backend.

Everything the LLM layer needs to know about which provider/model to use
lives here, so swapping Ollama for a cloud model later (per the project
spec) means changing this file / env vars, not the agent logic.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    frontend_origin: str = "http://localhost:5173"

    # Request timeout for talking to the LLM (local models can be slow on first load)
    llm_timeout_seconds: float = 120.0

    # Razorpay Test Mode. Leave both empty to run against the built-in stub
    # payment gateway (razorpay_client.py) - useful before you have real
    # test keys, or for demoing the Guardian's ALLOW/DENY logic without
    # touching Razorpay at all. Fill both in once you generate Test Mode
    # keys (Dashboard -> Test Mode -> Account & Settings -> API Keys) and
    # the backend switches to the real API automatically, no code changes.
    razorpay_key_id: str = ""
    razorpay_key_secret: str = ""


settings = Settings()
