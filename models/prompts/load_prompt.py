import os
import yaml
from typing import List, Dict, Any

# Folder containing this file (i.e., the prompts directory)
PROMPTS_DIR = os.path.dirname(__file__)

def list_prompts() -> List[str]:
    """Return prompt set names (filenames with extension)."""
    try:
        entries = os.listdir(PROMPTS_DIR)
    except FileNotFoundError:
        return []
    names = []
    for name in entries:
        if name.lower().endswith((".yaml", ".yml")):
            names.append(name)
    names.sort()
    return names

# Name expects yaml or yml suffix
def load_prompt(name: str) -> Dict[str, Any]: 
    """Load the whole YAML file as a Python dict."""
    path = os.path.join(PROMPTS_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such prompt file: {os.path.basename(path)}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {"prompt": "", "max_tokens": 0, "temperature": 0.0}
    return data["prompt"], data["max_tokens"], data["temperature"]
