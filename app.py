from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from model import AnimalCNN

app = Flask(__name__)

# load the model once when the app starts
# so we dont reload it every time someone uploads an image
model = AnimalCNN()
model.load_state_dict(torch.load('model/best_model.pth', map_location='cpu', weights_only=False))
model.eval()

class_names = ['bird', 'cat', 'dog']

# same transforms used during validation
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010]
    )
])

# folder to save uploaded images so we can display them on the page
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# main page - handles both showing the form (GET) and processing uploads (POST)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']

        if file:
            # save the uploaded file so we can show it back to the user
            # replace spaces with underscores so the image URL doesnt break
            filename = file.filename.replace(' ', '_')
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            # preprocess the image the same way we did during training
            image = Image.open(save_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0)

            # run the image through the model
            with torch.no_grad():
                outputs = model(image_tensor)

            # convert raw scores to probabilities
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, 1)

            predicted_class = class_names[predicted_index.item()]
            confidence_percent = round(confidence.item() * 100, 1)

            return render_template(
                'index.html',
                prediction=predicted_class,
                confidence=confidence_percent,
                image_url='uploads/' + filename
            )

    # GET request - just show the empty upload form
    return render_template('index.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
