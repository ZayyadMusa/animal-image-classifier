# Animal Image Classifier

This is my third project for my MSc AI portfolio. I built an image classifier that can tell the difference between cats, dogs and birds using a Convolutional Neural Network in PyTorch.

I trained two versions — one I built from scratch and one using a pretrained ResNet18 model. The ResNet18 version ended up being much more accurate which taught me a lot about transfer learning.

---

## What it does

You upload a photo of an animal and the app tells you whether it thinks it is a cat, a dog or a bird, along with how confident it is.

---

## My other projects

- [Sentiment Analyser](https://github.com/zayyadm65/sentiment-analyser)
- [House Price Predictor](https://github.com/zayyadm65/house-price-predictor)

---

## Models I trained

| Model | Test Accuracy |
|---|---|
| My CNN (from scratch) | TBC after training |
| ResNet18 (pretrained) | TBC after training |

---

## How to run it locally

1. Clone the repo
2. Create a virtual environment and activate it
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Train the model (or use the saved one in `/model`):
   ```
   python train.py
   ```
5. Run the web app:
   ```
   python app.py
   ```
6. Open your browser and go to `http://localhost:5000`

---

## Dataset

I used CIFAR-10 but only kept the three animal classes:
- Bird (class 2)
- Cat (class 3)
- Dog (class 5)

---

## Project structure

```
animal-image-classifier/
├── notebooks/         # Jupyter notebook with training and evaluation
├── model/             # Saved model weights
├── templates/         # HTML for the Flask app
├── static/            # CSS and saved plots
├── app.py             # Flask web app
├── model.py           # CNN architecture
├── train.py           # Training script
├── predict.py         # Prediction script
└── requirements.txt
```

---

## Screenshots

*(Coming soon — will add after the app is running)*

---

*Built as part of my MSc in Artificial Intelligence.*
