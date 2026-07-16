# Animal Image Classifier

This is my third project for my MSc AI portfolio. I built an image classifier that can tell the difference between cats, dogs and birds using a Convolutional Neural Network in PyTorch.

I trained two versions — one I designed from scratch and one using a pretrained ResNet18 model. The ResNet18 version ended up being significantly more accurate, which taught me a lot about why transfer learning is so powerful even for a simple task like this.

---

## Results

I trained both models on the same data (CIFAR-10 filtered to 3 classes) and compared them on the same test set.

| Model | Test Accuracy |
| --- | --- |
| My CNN (built from scratch) | update after training |
| ResNet18 (pretrained on ImageNet) | update after training |

The ResNet18 model did much better even though I only retrained the final layer. The pretrained weights already understood things like textures and shapes, so it had a big head start on a task like recognising animals.

My from-scratch CNN was still decent but clearly had less to work with since it had to learn everything from 32x32 images of only 15,000 training examples.

---

## What it does

You upload a photo of an animal and the app tells you whether it thinks it is a bird, a cat, or a dog — along with a confidence percentage.

---

## Dataset

I used the CIFAR-10 dataset but filtered it down to just three classes:

| Class | CIFAR-10 label |
| --- | --- |
| Bird | 2 |
| Cat | 3 |
| Dog | 5 |

This gives about 15,000 training images and 3,000 test images across the three classes. The images are only 32x32 pixels which makes it a harder task than it sounds — even humans struggle with some of them.

I split the training data 80/20 into a training set and a validation set, keeping the original test set separate for final evaluation.

---

## Models

### My CNN (from scratch)

I built a three-layer convolutional network with batch normalisation and dropout. The architecture is:

```text
Input (3 × 32 × 32)
→ Conv(32) + BatchNorm + ReLU + MaxPool
→ Conv(64) + BatchNorm + ReLU + MaxPool
→ Conv(128) + BatchNorm + ReLU
→ AdaptiveAvgPool
→ Linear(256) + ReLU + Dropout(0.5)
→ Linear(3)
```

### ResNet18 (transfer learning)

I loaded a ResNet18 model that was already trained on ImageNet (1.2 million images, 1000 classes). I froze all the existing layers and only trained the final fully connected layer to output 3 classes instead of 1000. This meant training only about 1,500 parameters instead of 11 million.

---

## How to run it locally

### 1. Clone the repo

```bash
git clone https://github.com/ZayyadMusa/animal-image-classifier.git
cd animal-image-classifier
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

This downloads CIFAR-10 automatically the first time (~160MB).

```bash
python train.py
```

Training takes around 5–10 minutes on a CPU. The best model is saved to `model/best_model.pth`.

### 5. Run the web app

```bash
python app.py
```

Then open your browser and go to `http://localhost:5000`

### 6. Or run a quick prediction from the command line

```bash
python predict.py path/to/your/image.jpg
```

---

## Project structure

```text
animal-image-classifier/
├── notebooks/
│   └── cnn_training.ipynb   training, evaluation, and comparison
├── model/
│   ├── best_model.pth        saved CNN weights (after training)
│   └── resnet18_best.pth     saved ResNet18 weights (after training)
├── templates/
│   └── index.html            Flask HTML template
├── static/
│   ├── style.css             app styling
│   ├── training_curves.png   loss and accuracy plots
│   ├── confusion_matrix.png  evaluation confusion matrix
│   └── model_comparison.png  CNN vs ResNet18 comparison chart
├── app.py                    Flask web app
├── model.py                  CNN and ResNet18 definitions
├── train.py                  training script
├── predict.py                single image prediction
└── requirements.txt
```

---

## What I learned

A few things that surprised me while building this:

- **BatchNorm made a big difference.** I tried without it early on and the training was much less stable.
- **Transfer learning is almost unfair.** Fine-tuning a frozen ResNet18 converges in far fewer epochs and ends up more accurate — even though we are only training one layer.
- **CIFAR-10 images are really small.** At 32×32 pixels some of the cat and dog images look almost identical even to me. The model does struggle with those the most (you can see it in the confusion matrix).
- **Early stopping helped.** Without it the model started memorising the training data after around epoch 12 or 13.

---


---

