import torch
import clip
from PIL import Image

device = "cpu" if torch.cuda.is_available() else "cuda"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("Dependency_tree.gv.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(["tree diagram", "a dog", "a cat", "tree"]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299