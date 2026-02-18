This repository has deep learning models for the classification and segmentation of medical images (chest X-rays). The main goal is to detect diseases from X-ray scans and find the exact locations of the lungs using advanced neural networks.

I developed the models and optimization methods in this project using my knowledge from the DeepLearning.AI specialization on Coursera. I especially focused on following courses:
- Neural Networks and Deep Learning
- Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
- Convolutional Neural Networks


Project Structure
The project is divided into two parts to solve different medical imaging problems:

1. Classification: In this part, the goal is to classify X-ray images into categories (Normal vs. Pneumonia).

denseNet.ipynb: After reading the paper "Densely Connected Convolutional Networks" (Huang et al.) i wanted to try DenseNet architecture. In DenseNet, each layer is directly connected to every other layer. This feature reuse makes it more efficient for medical images where every small detail is critical.

Inception.ipynb: I wanted to try this model after reading the "Going Deeper with Convolutions". I used the upgraded InceptionV3 model to get better results.. Lung diseases appear in very different sizes on an X-ray and Inception's ability to extract features at multiple scales simultaneously makes it a perfect choice for this problem.

3. Segmentation
In this part the goal is creating pixel level masks to show the exact location of lungs.

AttentionUNet.ipynb: I wanted to use the U-Net model, which is a classic industry standard for medical image segmentation. Then i decided to try improved version with the attention mechanism for better results. Attention mechanism helps the model suppress the noise from backgorund and focus on important areas.


Datasets
Classification Dataset: kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
Segmentation Dataset: kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels


How to Use
Clone this repository to your local computer using: 
git clone https://github.com/your-username/Chest-XRay-Analysis.git

Install the required Python libraries:
PyTorch
NumPy
Pandas
Matplotlib & Seaborn
OpenCV & Pillow
scikit-learn

Download the datasets and update the folder paths in the Jupyter Notebooks.

I uploaded the best performing model weights as .zip files in the Releases section. If you download the pre-trained weights, please rename the .pth files to match the paths expected in the notebooks.
