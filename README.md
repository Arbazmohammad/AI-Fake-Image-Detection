# AI-Fake-Image-Detection

# Introduction

Our approach involves generating images as a first step, we considered a Celeb-DF dataset, which contains over 6000 real or fake videos, and from these videos, we have extracted images at an interval of 3 seconds using OpenCV and generated face cropped images using  MTCNN with the help of in-built functions named VideoCapture and imwrite to process each frame and extracted a .png file from .mp4 file. Second, we performed some analysis to make images ready to ingest them for training so as part of preprocessing we resized the image to 128x128 to make sure all images have the same dimensions also we labeled the real images and fake images as 0 and 1 using encoding to label the images and split the images and its labels into training and testing data of 80% and 20%. 

In the final stages, the prepared training and testing datasets were fed into the model through an Image Data Generator, which normalizes pixel values and applies transformations like rotation and flipping to enhance the training process. After training, the model was used to evaluate the preprocessed validation data to determine the authenticity of the images, classifying them as either real or fake. This paper outlines our comprehensive approach to addressing the challenges posed by deep fakes in the realm of digital media.

# Flow Diagram

![DFD (1)](https://github.com/schebrolu6405/Deep_fake_video_detection/assets/143303673/f718580d-ce69-4e8b-ab96-880d7544efb8)

# Analysis

Our primary objective was to evaluate the efficacy of machine learning algorithms in the detection of deep fake images, with a focus on comparing various models' performance against a baseline. For our baseline model, we chose the ResNet50 architecture, a robust convolutional neural network that has demonstrated high effectiveness in various image recognition tasks. ResNet50 was selected due to its deep residual learning framework, which helps in training deeper networks by addressing the vanishing gradient problem. This makes it an ideal starting point for our comparisons.

The data handling and preparation phase was critical for ensuring the consistency and quality of the inputs fed into our machine learning models. Images, extracted as the first frame from video files using MTCNN, were batched and preprocessed uniformly to maintain consistency across the dataset. These images were resized to a standard dimension of 128x128 pixels, converted to RGB color space, and normalized using the preprocessing function specific to ResNet50. Data augmentation techniques such as rotation, width, height shifting, and horizontal flipping were applied using the ImageDataGenerator from TensorFlow's Keras API to enhance the model's generalization capabilities. For model training and validation, the dataset was split into 80% for training and 20% for testing, with images batched in sizes of 32 during model training to optimize memory usage and computational efficiency.

Hyperparameter tuning was an essential part of refining our models. For the ResNet50 model, the learning rate, number of epochs, and batch size were the primary hyperparameters adjusted. The learning rate was initially set to 0.0001, utilizing an Adam optimizer to facilitate efficient convergence. The model was trained over 10 epochs to balance between adequate learning and computational efficiency. The performance evaluation of our models was systematically carried out by comparing accuracy and loss metrics on the validation set. Binary cross-entropy was used as the loss function, which is suitable for the binary classification tasks at hand. Additionally, the accuracy metric provided a direct measure of model performance in correctly classifying the images as real or fake.

In conclusion, the analytical approach adopted in this research provided a detailed examination of the effectiveness of the baseline model and facilitated a structured comparison with other machine learning algorithms. The rigorous data preprocessing, systematic hyperparameter tuning and comprehensive performance evaluation established a robust framework for assessing the capabilities of deep learning models in the context of deep fake detection. This methodological rigor ensures that the findings are reliable and can serve as a benchmark for future research in the field.

# Results
![output](https://github.com/schebrolu6405/Deep_fake_video_detection/assets/143303673/984738ed-def7-4233-ab5b-2679340c7acd)

![lossoutput](https://github.com/schebrolu6405/Deep_fake_video_detection/assets/143303673/5ee7cc23-4792-44a2-a56c-c30a109197d6)

![conf_matrix](https://github.com/schebrolu6405/Deep_fake_video_detection/assets/143303673/78a3a57a-f5ff-4574-aa15-2e714ba98102)


