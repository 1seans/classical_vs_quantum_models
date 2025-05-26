# replay.py

import json
import os
from datetime import datetime
import uuid

HISTORY_FILE = "data/history.json"

def save_run(model_name, params, summary, raw_data):
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "params": params,
        "summary": summary,
        "data": raw_data  # Optional: Final prices, avg path, full matrix
    }

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.insert(0, entry)  # Most recent first
    history = history[:10]    # Keep only last 10 runs

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    return []
