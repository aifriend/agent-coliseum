import os
import sys
from pathlib import Path

# Dummy values so importing the agent modules never needs real credentials.
# No test calls the model; anything that would is replaced by a stub.
os.environ.setdefault("AZURE_API_KEY", "test-key")
os.environ.setdefault("AZURE_BASE_URL", "http://localhost:0/openai/v1/")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
