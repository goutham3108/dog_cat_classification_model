Cat vs Dog Image Classification using Transfer Learning (MobileNetV2)
Overview
This project implements a Cat vs Dog Image Classification Model using Transfer Learning with the MobileNetV2 architecture.
It demonstrates a complete end-to-end deep learning workflow—covering dataset preparation, preprocessing, model training, fine-tuning, evaluation, and deployment.
The model is lightweight, accurate, and designed for efficient training on both CPU and GPU systems.
A live Streamlit web application is available for testing the model:
Try the deployed model here →

Objectives


Design an end-to-end deep learning pipeline for dataset preparation, model training, and deployment.


Apply Transfer Learning using MobileNetV2 to achieve high accuracy and reduce training time.


Automate the data cleaning, preprocessing, and augmentation process.


Build a deployable model accessible through a web interface.



Features


Automated dataset download, extraction, and structuring


Image preprocessing and augmentation


Transfer learning with MobileNetV2 (ImageNet weights)


Two-phase training: Feature Extraction and Fine-Tuning


GPU acceleration support (if available)


Model evaluation and saving in .keras format


Ready for deployment using Streamlit



Project Structure
├── train_model.py                # Main training script (end-to-end pipeline)
├── cat_dog_classifier.keras      # Trained model file
├── app.py                        # Streamlit web application (deployment)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation


Workflow
1. Dataset Preparation


Downloads the Oxford-IIIT Pet Dataset


Extracts and organizes images into cat/ and dog/ folders


Removes corrupted or missing files


Splits data into 80% training and 20% validation subsets


2. Data Preprocessing


Resizes all images to 160×160×3


Converts images to RGB and normalized pixel values to the range [-1, 1]


Applies random data augmentation (flip, rotation, zoom)


Uses TensorFlow’s AUTOTUNE for optimized data loading


3. Model Architecture


Base Model: MobileNetV2 (pre-trained on ImageNet)


Custom Layers Added:


GlobalAveragePooling2D()


Dropout(0.2)


Dense(1, activation='sigmoid')




4. Training Process


Phase 1: Feature Extraction


Base model frozen


Only top custom layers trained for 10 epochs




Phase 2: Fine-Tuning


Last 55 layers of base model unfrozen


Trained with smaller learning rate for 10 additional epochs




5. Evaluation


Validation accuracy consistently between 96%–98%


Model saved as cat_dog_classifier.keras for deployment



Installation and Setup
Prerequisites


Python 3.9 or higher


TensorFlow 2.10 or higher


NVIDIA GPU with CUDA (optional but recommended)


Step 1: Clone the Repository
git clone https://github.com/<your-username>/cat-vs-dog-classifier.git
cd cat-vs-dog-classifier

Step 2: Create and Activate a Virtual Environment
python -m venv env
env\Scripts\activate       # Windows
# or
source env/bin/activate    # macOS/Linux

Step 3: Install Dependencies
pip install -r requirements.txt

Step 4: Train the Model
python train_model.py

Step 5: Run the Web Application
streamlit run app.py

Or visit the deployed version directly:
https://dogcatclassificationmodel.streamlit.app/

Software Requirements
LibraryPurposeTensorFlow / KerasCore deep learning framework for model training and transfer learningNumPyNumerical array manipulation and tensor operationsPillow (PIL)Image input/output and resizingOpenCVAdditional image processing utilitiesMatplotlibVisualization of training metricsStreamlitWeb-based deployment interface

Model Summary
LayerDescriptionMobileNetV2 (Base)Pre-trained CNN model on ImageNet, frozen for feature extractionGlobalAveragePooling2DReduces spatial dimensions and prevents overfittingDropout (0.2)Regularization layer to improve generalizationDense (1, Sigmoid)Outputs binary prediction for cat/dog classification
Total Parameters: ~2.3 million
Trainable Parameters (after fine-tuning): ~1.1 million

Results
MetricTrainingValidationAccuracy97%96%Loss0.120.15
Observations:


Rapid convergence due to transfer learning


Low overfitting due to dropout and augmentation


Efficient and lightweight model suitable for real-time inference



Deployment
The trained model is deployed using Streamlit and hosted at:
https://dogcatclassificationmodel.streamlit.app/
The application allows users to:


Upload an image of a cat or dog


View prediction result and confidence score


Experience real-time inference directly in the browser



Key Takeaways


Transfer Learning using MobileNetV2 significantly reduces training time and data requirements.


The GlobalAveragePooling2D layer minimizes parameters, improving efficiency.


Data augmentation and dropout prevent overfitting and improve generalization.


Using TensorFlow’s AUTOTUNE optimizes the input pipeline for maximum GPU utilization.


The model achieves state-of-the-art performance while remaining lightweight and deployable.



References


Oxford-IIIT Pet Dataset: http://www.robots.ox.ac.uk/~vgg/data/pets


TensorFlow Documentation: https://www.tensorflow.org


MobileNetV2 Paper: Sandler et al., “MobileNetV2: Inverted Residuals and Linear Bottlenecks” (2018)



Would you like me to also generate a clean requirements.txt file (matching your project dependencies precisely for GitHub and Streamlit deployment)?
