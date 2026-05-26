import os
from pathlib import Path

from dotenv import load_dotenv

# Load amia/.env regardless of cwd when running `python -m backend.*`
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
def get_llm(temperature: float = 0.1, max_tokens: int = 4000):
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    model = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. Add it to amia/.env or export it in your shell."
            )
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to amia/.env or export it in your shell."
            )
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
        )
    else:
        raise ValueError(f"Invalid provider: {provider}")


def get_fast_llm():
    """
    Returns a fast, cheap LLM for classification and routing tasks.
    Uses Claude Haiku or GPT-4o-mini — much faster and cheaper than
    the full models, but lower quality for complex tasks.
    """
    provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model="claude-3-haiku-20240307",
            temperature=0.0,
            max_tokens=512,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=512,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_embedding_model(use_local: bool = True):
    if use_local:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


__all__ = ["get_llm", "get_fast_llm", "get_embedding_model"]