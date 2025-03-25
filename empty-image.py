from PIL import Image
import json
import base64
from io import BytesIO

# Create a new 28x28 RGB image (black by default)
img = Image.new('RGB', (28, 28))

# Convert the image to base64
buffer = BytesIO()
img.save(buffer, format='PNG')
img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

# Create a dictionary with the base64 data
img_data = {
    'image': img_base64
}

# Save to JSON file
with open('empty_image.json', 'w') as f:
    json.dump(img_data, f)
