import cv2
import numpy as np

def process_image(file):
   nparr = np.frombuffer(file.read(), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   
   # Example processing: Convert to grayscale
   gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
   
   # Additional processing options
   resized_image = cv2.resize(gray_image, (256, 256))  # Resize to 256x256
   filtered_image = cv2.GaussianBlur(resized_image, (5, 5), 0)  # Apply Gaussian blur
   
   return {
       "shape": filtered_image.shape,
       "status": "Processed to grayscale and resized",
       "data": filtered_image.tolist()  # Convert to list for JSON serialization
   }
