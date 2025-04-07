# 🌸 Image Classification with Transfer Learning (PyTorch)

This project performs image classification on a flower dataset using **transfer learning** with pretrained CNN architectures like VGG16. It includes a Jupyter notebook for development and two command-line scripts: one for training a model and another for making predictions.

---

## 📁 Project Structure

```bash
image-classification-project/
│
├── notebooks/
│   └── development_notebook.ipynb      # Notebook with step-by-step development
│
├── train.py                            # CLI script to train the model
├── predict.py                          # CLI script to predict image classes
├── checkpoint.pth                      # Saved model checkpoint
├── cat_to_name.json                    # Class label to flower name mapping
│
├── utils/
│   └── model_utils.py                  # Utility functions (load model, process image, etc.)
│
├── requirements.txt
├── .gitignore
└── README.md

🚀 Features
📦 Data loading using torchvision.datasets.ImageFolder

🔄 Data augmentation (rotation, flip, crop) using transforms

🧠 Transfer Learning with pretrained models (vgg16, resnet18, etc.)

🔧 Custom classifier with configurable hyperparameters

🔥 GPU support for faster training and inference

💾 Save and load trained model checkpoints

🖼️ Predict top K classes from an image

📈 Visualize predictions using matplotlib

⚙️ Installation
1.Clone the repo:
git clone https://github.com/your-username/image-classification-project.git
cd image-classification-project

2.Install dependencies:
pip install -r requirements.txt

(Optional) Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

🧪 Development Notebook
Run the notebook for an interactive walkthrough:
jupyter notebook notebooks/development_notebook.ipynb

🛠️ Train a Model
python train.py --data_dir data --save_dir checkpoints --arch vgg16 \
--learning_rate 0.003 --hidden_units 512 --epochs 5 --gpu
🔧 Options:
Argument	Description
--data_dir	Path to dataset directory
--save_dir	Directory to save model checkpoint
--arch	Model architecture (vgg16, resnet18)
--learning_rate	Learning rate for optimizer
--hidden_units	Hidden units in classifier
--epochs	Number of training epochs
--gpu	Use GPU if available

🔍 Predict an Image
python predict.py image.jpg checkpoint.pth --top_k 5 \
--category_names cat_to_name.json --gpu
🔧 Options:
Argument	Description
image.jpg	Path to input image
checkpoint.pth	Path to trained model checkpoint
--top_k	Return top K most probable classes
--category_names	JSON file mapping class indices to names
--gpu	Use GPU for inference (if available)

🌸 Sample Prediction Output
Top 5 Predictions:
1. Tulip (94.3%)
2. Rose (3.2%)
3. Daffodil (1.5%)
...

✨ Acknowledgements
Dataset: 102 Category Flower Dataset

PyTorch & torchvision

Udacity Deep Learning Nanodegree inspiration
