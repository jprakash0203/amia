import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def get_embedding_model(use_local: bool = True):
    if use_local:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name = "all-MiniLM-L6-v2",
            model_kwargs = {"device": "cpu"},
            encode_kwargs = {"device": "cpu"}
        )

    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


if __name__ == "__main__":
    model = get_embedding_model(use_local=True)
    vec = model.embed_query("test sentence about AI")
    print(f"Vector length: {len(vec)}")   # 384
    print(f"First 5 values: {vec[:5]}")
    print("✅ embeddings.py working")