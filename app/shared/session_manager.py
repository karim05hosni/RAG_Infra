session_store: dict[str, list] = {}

def session_exists(session_id: str) -> bool:
    return session_id in session_store

def create_session(session_id: str):
    session_store[session_id] = {"history": []}
    
def get_session_history(session_id: str) -> list:
    if session_exists(session_id):
        return session_store[session_id]["history"]
    else:
        raise ValueError(f"Session '{session_id}' does not exist.")

def add_to_session_history(session_id: str, step: dict):
    if session_exists(session_id):
        session_store[session_id]["history"].append(step)
    else:
        raise ValueError(f"Session '{session_id}' does not exist.")