from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Take 3 sample RGB values and calculate the cosine similarity between them

# Sample RGB values
rgb1 = [255, 0, 0]  # Red
rgb2 = [0, 255, 0]  # Green
rgb3 = [0, 0, 255]  # Blue

# Convert to unit vectors
rgb1 = rgb1 / np.linalg.norm(rgb1)
rgb2 = rgb2 / np.linalg.norm(rgb2)
rgb3 = rgb3 / np.linalg.norm(rgb3)

print (rgb1)
print (rgb2)
print (rgb3)

# Print the cosine similarity between the vectors
print(cosine_similarity([rgb1], [rgb2]))
print(cosine_similarity([rgb2], [rgb3]))
print(cosine_similarity([rgb3], [rgb1]))