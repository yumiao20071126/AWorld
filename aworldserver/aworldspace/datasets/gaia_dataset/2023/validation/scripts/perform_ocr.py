from PIL import Image
import pytesseract

# Path to the image file
image_path = '/Users/yuchengyue/AWorld_mcp/gaia/gaia_dataset/2023/validation/5b2a14e8-6e59-479c-80e3-4696e8980152.jpg'

# Open the image using PIL
img = Image.open(image_path)

# Extract text from the image using Tesseract
extracted_text = pytesseract.image_to_string(img)

# Print the extracted text
print(extracted_text)