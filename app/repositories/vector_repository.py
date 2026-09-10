
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct, Modifier, VectorParams, SparseVectorParams, Distance, Document

load_dotenv()  # Load environment variables from .env file
# Connect to your local Docker container
qdrant_client = QdrantClient(url=os.getenv('qdrant_url')) 
collection_name = "my_rag"

def delete_all_collections():
    collections = qdrant_client.get_collections().collections
    for collection in collections:
        print(f"Deleting collection: {collection.name}")
        qdrant_client.delete_collection(collection.name)
# delete_all_collections()

if not qdrant_client.collection_exists(collection_name):
    print(f"Creating collection '{collection_name}' with vector size 384 .")
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.DOT)
    )


# You can now use all standard client operations (create_collection, upsert, query_points)
print(qdrant_client.get_collections())


def qdrant_search(query_vector, top_k=20):
    search_result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,          # Pass the list of floats directly here
        limit=top_k,                         # Number of closest results to return
        with_payload=True,               # Ensure the text metadata comes back
    )
    return search_result

def add_to_qdrant(grouped_chunks):
    vector_size = len(grouped_chunks[0][0]['vector'])
    points = []
    print(f"Upserting {len(grouped_chunks)} groups of chunks to Qdrant.")
    for group in grouped_chunks:
        for chunk in group:
            points.append(PointStruct(id=chunk['chunk_id'], vector=chunk['vector'], payload={"source_id": chunk['source_id'], "chunk_id": chunk['chunk_id']}))
        # debug points byte size
        print(f"Debug: {len(points)} points to upsert.")
        qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        points = []
