from dotenv import load_dotenv
import os

load_dotenv()


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
