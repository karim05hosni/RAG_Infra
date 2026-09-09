
import json
from uuid import UUID
import uuid

def generate_chunk_id(doc_id, chunk_index):
    return uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}__chunk_{chunk_index}")

def generate_doc_id(file_path):
    return uuid.uuid5(uuid.NAMESPACE_DNS, file_path)

def uuid_convert(obj):
    """Converts UUID objects to strings for JSON serialization."""
    if isinstance(obj, UUID):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def split_list_by_max_bytes(lst, max_bytes):
    grouped_chunks = []
    current_group = []
    current_size = 0

    for item in lst:
        # Pass the custom converter to the default parameter
        item_json = json.dumps(item, default=uuid_convert)
        item_size = len(item_json.encode('utf-8'))
        
        if current_size + item_size > max_bytes:
            if current_group: # Prevent appending empty lists if a single item is large
                grouped_chunks.append(current_group)
            current_group = [item]
            current_size = item_size
        else:
            current_group.append(item)
            current_size += item_size

    if current_group:
        grouped_chunks.append(current_group)

    return grouped_chunks