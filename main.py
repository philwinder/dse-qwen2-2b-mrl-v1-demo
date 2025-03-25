import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

# Add a progress callback to see what's happening
from transformers.utils import logging
logging.set_verbosity_debug()

min_pixels = 1*28*28
max_pixels = 2560*28*28

# Clear CUDA cache first
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Load processor
processor = AutoProcessor.from_pretrained(
    "MrLight/dse-qwen2-2b-mrl-v1", 
)

# Load model with minimal configurations first
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "MrLight/dse-qwen2-2b-mrl-v1", 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2",
)

# Move to GPU after loading
if torch.cuda.is_available():
    model = model.cuda()
model = model.eval()

processor.tokenizer.padding_side = "left"
model.padding_side = "left"

def get_embedding(last_hidden_state: torch.Tensor, dimension: int) -> torch.Tensor:
    reps = last_hidden_state[:, -1]
    reps = torch.nn.functional.normalize(reps[:, :dimension], p=2, dim=-1)
    return reps

print(f"Model loaded on {model.device}")


from PIL import Image
queries = ["panda", "cat", "bear"]
query_messages = []
for query in queries:
    message = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': Image.new('RGB', (28, 28))}, # need a dummy image here for an easier process.
                {'type': 'text', 'text': f'Query: {query}'},
            ]
        }
    ]
    query_messages.append(message)
query_texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
    for msg in query_messages
]
query_image_inputs, query_video_inputs = process_vision_info(query_messages)
query_inputs = processor(text=query_texts, images=query_image_inputs, videos=query_video_inputs, padding='longest', return_tensors='pt').to(model.device)
cache_position = torch.arange(0, len(query_texts))
query_inputs = model.prepare_inputs_for_generation(**query_inputs, cache_position=cache_position, use_cache=False)
with torch.no_grad():
  output = model(**query_inputs, return_dict=True, output_hidden_states=True)
query_embeddings = get_embedding(output.hidden_states[-1], 1536) # adjust dimensionality for efficiency trade-off, e.g. 512

print(query_embeddings)


import requests
from io import BytesIO

# URLs of the images
url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Mei_Xiang_Female_Panda_Eating_Bamboo_While_Lying_Down_33.jpg/800px-Mei_Xiang_Female_Panda_Eating_Bamboo_While_Lying_Down_33.jpg"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Brown_tabby_cat_2018_G1.jpg/792px-Brown_tabby_cat_2018_G1.jpg"


# Download and open images with error handling
def load_image_from_url(url):
    try:
        headers = {'User-Agent': 'CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        return Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        # Return a blank image as fallback
        return Image.new('RGB', (224, 224))

doc_image1 = load_image_from_url(url1)
doc_image2 = load_image_from_url(url2)

doc_images = [doc_image1, doc_image2]
doc_messages = []
for doc in doc_images:
    message = [
        {
            'role': 'user',
            'content': [
                {'type': 'image', 'image': doc}, #'resized_height':680 , 'resized_width':680} # adjust the image size for efficiency trade-off
                {'type': 'text', 'text': 'What is shown in this image?'}
            ]
        }
    ]
    doc_messages.append(message)
doc_texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
    for msg in doc_messages
]
doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
doc_inputs = processor(text=doc_texts, images=doc_image_inputs, videos=doc_video_inputs, padding='longest', return_tensors='pt').to(model.device)
cache_position = torch.arange(0, len(doc_texts))
doc_inputs = model.prepare_inputs_for_generation(**doc_inputs, cache_position=cache_position, use_cache=False)
with torch.no_grad():
    output = model(**doc_inputs, return_dict=True, output_hidden_states=True)
doc_embeddings = get_embedding(output.hidden_states[-1], 1536) # adjust dimensionality for efficiency trade-off e.g. 512

# Output the embeddings to a json file
import json

with open('embeddings.json', 'w') as f:
    json.dump({
        'query_embeddings': query_embeddings.cpu().float().numpy().tolist(),
        'doc_embeddings': doc_embeddings.cpu().float().numpy().tolist()
    }, f)

from torch.nn.functional import cosine_similarity
num_queries = query_embeddings.size(0)
num_passages = doc_embeddings.size(0)

for i in range(num_queries):
    query_embedding = query_embeddings[i].unsqueeze(0)
    similarities = cosine_similarity(query_embedding, doc_embeddings)
    print(f"Similarities for Query {i+1}: {similarities.cpu().float().numpy()}")

