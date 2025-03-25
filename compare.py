import torch
from torch.nn.functional import cosine_similarity
import json

def load_embedding(file_path, selector=None):
    """
    Load embedding vector from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing the embedding
        selector: Optional selector path (e.g., 'doc_embeddings.0' or 'key1.key2')
    
    Returns:
        torch.Tensor: Loaded embedding vector
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle nested selectors
    if selector is not None:
        for key in selector.split('.'):
            if isinstance(data, dict):
                if key not in data:
                    raise KeyError(f"Key '{key}' not found in {file_path}")
                data = data[key]
            elif isinstance(data, list):
                try:
                    idx = int(key)
                    data = data[idx]
                except (ValueError, IndexError):
                    raise ValueError(f"Invalid index '{key}' for list in {file_path}")
            else:
                raise ValueError(f"Cannot select '{key}' from non-dict/list data")
    
    return torch.tensor(data)

def compare_embeddings(vector1, vector2):
    """
    Calculate cosine similarity between two vectors using PyTorch.
    
    Args:
        vector1: First embedding vector (torch tensor)
        vector2: Second embedding vector (torch tensor)
    
    Returns:
        float: Cosine similarity score between -1 and 1
    """
    # Ensure inputs are PyTorch tensors and reshape if needed
    if not isinstance(vector1, torch.Tensor):
        vector1 = torch.tensor(vector1)
    if not isinstance(vector2, torch.Tensor):
        vector2 = torch.tensor(vector2)
    
    # Add batch dimension if needed
    if vector1.dim() == 1:
        vector1 = vector1.unsqueeze(0)
    if vector2.dim() == 1:
        vector2 = vector2.unsqueeze(0)
    
    # Calculate similarity
    similarity = cosine_similarity(vector1, vector2, dim=1)
    
    return similarity.item()

# Example usage
if __name__ == "__main__":
    try:
        # Panda
        panda_python_embedding = load_embedding('embeddings.json', selector='doc_embeddings.0')
        panda_python_query = load_embedding('embeddings.json', selector='query_embeddings.0')
        panda_vllm_embedding = load_embedding('panda.json', selector='embedding')
        panda_vllm_query = load_embedding('panda.json', selector='query')

        # Check sizes
        print(f"panda_python_embedding size: {panda_python_embedding.size()}")
        print(f"panda_vllm_embedding size: {panda_vllm_embedding.size()}")
        print(f"panda_python_query size: {panda_python_query.size()}")
        print(f"panda_vllm_query size: {panda_vllm_query.size()}")

        similarity_score = compare_embeddings(panda_python_embedding, panda_vllm_embedding)
        print(f"Image similarity panda-panda (should be 1): {similarity_score:.4f}")
        similarity_score = compare_embeddings(panda_python_query, panda_vllm_query)
        print(f"Query similarity panda-panda (should be 1): {similarity_score:.4f}")
        similarity_score = compare_embeddings(panda_vllm_embedding, panda_vllm_query)
        print(f"VLLM query panda-panda (should be 0.5078125): {similarity_score:.4f}")

        # Cat
        cat_python_embedding = load_embedding('embeddings.json', selector='doc_embeddings.1')
        cat_python_query = load_embedding('embeddings.json', selector='query_embeddings.1')
        cat_vllm_embedding = load_embedding('cat.json', selector='embedding')
        cat_vllm_query = load_embedding('cat.json', selector='query')

        similarity_score = compare_embeddings(cat_python_embedding, cat_vllm_embedding)
        print(f"Image similarity cat-cat (should be 1): {similarity_score:.4f}")
        similarity_score = compare_embeddings(cat_python_query, cat_vllm_query)
        print(f"Query similarity cat-cat (should be 1): {similarity_score:.4f}")
        similarity_score = compare_embeddings(cat_vllm_embedding, cat_vllm_query)
        print(f"VLLM query cat (should be 0.4921875): {similarity_score:.4f}")

        # Bear
        bear_vllm_query = load_embedding('bear.json', selector='query')

        similarity_score = compare_embeddings(panda_vllm_embedding, cat_vllm_query)
        print(f"VLLM query panda-cat (should be 0.38671875): {similarity_score:.4f}")

        similarity_score = compare_embeddings(cat_vllm_embedding, panda_vllm_query)
        print(f"VLLM query cat-panda (should be 0.29296875): {similarity_score:.4f}")

        similarity_score = compare_embeddings(bear_vllm_query, panda_vllm_embedding)
        print(f"VLLM query bear-panda (should be 0.40429688): {similarity_score:.4f}")

        similarity_score = compare_embeddings(bear_vllm_query, cat_vllm_embedding)
        print(f"VLLM query bear-cat (should be 0.32421875): {similarity_score:.4f}")


        
    except (KeyError, ValueError) as e:
        print(f"Error: {e}")
