import os
from dotenv import load_dotenv
from langfuse import Langfuse

load_dotenv(override=True)

def get_langfuse_client():
    return Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY")
    )
