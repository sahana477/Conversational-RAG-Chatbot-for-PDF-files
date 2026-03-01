import os
from dotenv import load_dotenv
from langfuse import Langfuse



def get_langfuse():
    return Langfuse(
        public_key="pk-lf-8997c9aa-a51c-43b0-869a-573b1c1937ba",
        secret_key="sk-lf-bbdf272d-8c59-4c96-af2e-b7625025d9af",
        host="http://localhost:3000",
        debug=True
    )
