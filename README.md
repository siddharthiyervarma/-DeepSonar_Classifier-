## DeepSonar Classifier: Distinguishing Rocks from Metal Cylinders using Neural Networks

1. Introduction:
In the realm of underwater exploration, distinguishing between rocks and metal cylinders is crucial for various applications, from marine research to navigation. The dataset at hand provides a comprehensive description of sonar chirp returns, representing the strength of echoes at different angles. This binary classification problem seeks to develop a robust model capable of accurately identifying whether the sonar signals correspond to rocks or metal cylinders.

2. Objective:
The primary objective of this project is to design and train a deep neural network that can generalize well to new, unseen data, effectively distinguishing between rocks and metal cylinders based on the sonar chirp returns.

3. The DATA: 
The dataset consists of 60 input variables representing the strength of the returns at different angles. The dataset is loaded from a CSV file named "sonar.csv" using the Pandas library. It contains 61 columns, where the first 60 columns are input variables, and the last column (60) is the target variable indicating whether the object is a rock or a metal cylinder.
## 🛠 Skills
Data Preprocessing 

Deep Neural Networks 

Model Training 

Model Engeeneering 


## Roadmap: 
1. Data Preprocessing:
Label Encoding:
Convert the categorical target variable ("Rock" and "Mine") into numerical format using label encoding.
Train-Test Split:
Split the dataset into training and testing sets using the train_test_split function from Scikit-learn.
2. Initial Model Construction:
Neural Network Architecture:
Construct a neural network model using TensorFlow and Keras.
Layer Configuration:
Design the model with multiple dense layers, incorporating ReLU activation functions.
Output Layer:
Utilize a sigmoid activation function in the output layer for binary classification.
Model Compilation:
Compile the model with binary cross-entropy loss and the Adam optimizer.
3. Model Training:
Epochs and Batch Size:
Train the model on the training set for 100 epochs with a batch size of 8.
Validation Monitoring:
Monitor the model's performance on a validation set during training to detect overfitting.
4. Model Evaluation:
Performance Metrics:
Evaluate the model's performance on both the training and testing sets.
Metrics Display:
Calculate and display accuracy and loss metrics.
5. Handling Overfitting:
L2 Regularization:
Address overfitting by adding L2 regularization to the first dense layer.
Dropout Layers:
Introduce randomness and prevent overfitting by incorporating dropout layers.
Early Stopping:
Implement early stopping to halt training if the validation accuracy does not improve for a set number of epochs.
6. Model Checkpoint:
Save Best Weights:
Implement a ModelCheckpoint callback to save the model's weights only when the validation loss improves.
7. Load Best Model:
Load Weights:
Load the weights of the best model saved using the ModelCheckpoint callback.
Evaluation:
Evaluate the loaded model on the test set to ensure optimal performance.
8. Results:
Accuracy Display:
Highlight the final model's test accuracy, which achieves approximately 73.08%.
## Authors

- [@siddharthiyervarma](https://github.com/siddharthiyervarma)

