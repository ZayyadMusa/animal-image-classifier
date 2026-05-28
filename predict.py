# this script takes an image path and returns
# what animal the model thinks it is
# usage: python predict.py path/to/your/image.jpg

import torch
import torchvision.transforms as transforms
from PIL import Image
import sys

from model import AnimalCNN


# class names in the same order as the training labels
# bird=0, cat=1, dog=2
class_names = ['bird', 'cat', 'dog']

# hardcoded model path - this works but ideally would be a command line argument
model_path = 'model/best_model.pth'

# same transforms we used for validation - no random flipping or cropping
# resize to 32x32 because thats the size our model was trained on
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])


def load_model(path):
    model = AnimalCNN()
    # map_location='cpu' means it loads fine even if it was trained on a GPU
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False))
    model.eval()
    return model


def get_prediction(image_path, model):
    # open the image and make sure it has 3 colour channels (RGB)
    image = Image.open(image_path).convert('RGB')

    # apply the transforms so the image matches what the model expects
    image_tensor = transform(image)

    # the model expects a batch dimension at the front
    # unsqueeze(0) changes shape from (3, 32, 32) to (1, 3, 32, 32)
    image_tensor = image_tensor.unsqueeze(0)

    # run through the model - no_grad because we dont need gradients here
    with torch.no_grad():
        outputs = model(image_tensor)

    # softmax converts the raw scores into probabilities that add up to 1
    probabilities = torch.softmax(outputs, dim=1)

    # get the class index with the highest probability
    confidence, predicted_index = torch.max(probabilities, 1)

    predicted_class = class_names[predicted_index.item()]
    confidence_percent = confidence.item() * 100

    return predicted_class, confidence_percent


# --- run the prediction ---

if len(sys.argv) < 2:
    print('Please provide an image path.')
    print('Usage: python predict.py path/to/image.jpg')
    sys.exit(1)

image_path = sys.argv[1]

print(f'Loading model...')
model = load_model(model_path)

print(f'Analysing image: {image_path}')
predicted_class, confidence = get_prediction(image_path, model)

print()
print(f'I think this is a {predicted_class} ({confidence:.1f}% confident)')
