# **Capstone4: Machine Learning**

This repository contains a Google Colab notebook that walks through training various machine learning models from polynomial regression to deep neural networks by using Scikit-learn and Keras. This capstone is the final unit and represents the final learning outcomes of my DS201 course. 

### **Learning Objectives**

By using this notebook and the supplemntary tutorial video, you'll be able to understand:
- Why data splitting and scaling are critical before training models.
- The difference between **underfitting** and **overfitting**, and how to detect them by using performance metrics.
- How increasing **model complexity** (as is demonstrated in the polynomial degree section) can affect performance.
- How **Ridge (L2) regularization** helps mitigate overfitting in high-degree polynomial models.
- Why **deeper neural networks** generally outperform simpler ones.

### **Overview of the Notebook**

Part 1: Regression

1.1 Generate a synthetic dataset.
1.2 Split and scale the data.
1.3 Build linear regression models using polynomial features of increasing degree to observe underfitting versus overfitting.
1.4 Apply Ridge regularization to see its effect on an overfitting model.
Part 2: Classification

2.1 Load an image dataset for classification.
2.2 Split and scale the data.
2.3 Train a "simple" feedforward Neural Network.
2.4 Train a "deeper" feedforward Neural Network.
2.5 Show the performance of our models on two different datasets.

### **Visualizations**
1. This scatterplot is to visualize the synthetic dataset that we created to be able to train our machine learning model on. This is set to improve it's ability to recognize patterns within datasets and produce the desired results it is programmed to deliver. 


<img width="693" alt="image" src="https://github.com/user-attachments/assets/739a6ce4-f521-45fb-b9fe-7ced141112f1" />


2. After training each model, the performance (R² score) is computed on both the training and test sets to observe the effect of model complexity. The higher an R² score is, the better the model is. First, we had to build a linear model using degree 1 polynomial features (which is just your original data with an intercept).


<img width="538" alt="image" src="https://github.com/user-attachments/assets/79b46837-634e-498c-a0fa-9f26f4d16782" />


As you can see, the Polynomial Degree =1 intercept line doesn't cluster as closely with the data which means that while the data is represented by the intecept line the model suffers from underfitting. 

3. When you increase the model complexity, you provide the model with more flexibility by increasing the degree of the polynomial. As shown in the image provided, the degree of the polynomial has been increased to 5 in hopes of correcting the underfitting issue from the previous model.


<img width="541" alt="image" src="https://github.com/user-attachments/assets/fe47979b-8e46-46ce-85be-8ba030a479f6" />


A degree 5 polynomial allows the model to capture non-linear relationships. This can improve performance, but also increases the risk of overfitting. Overfitting can be detected by comparing training and testing errors. A large gap often means the model is memorizing the training data rather than generalizing. 

4. Finally, we pushed our model to compensate a Polynomial Degree = 20, which was made possible by using the Pipeline function.


<img width="542" alt="image" src="https://github.com/user-attachments/assets/441e29d5-60ce-467c-b143-38b569fc75b4" />


By using degree 20, our model is given extreme flexibility, which has the complexity to account for the differing data points but can almost certainly lead to overfitting unless a regularization is applied to reduce this in the model.

5. To prevent the overfitting in the high-complexity model previously shown, we utilized a Ridge regression, which adds an L2 penalty to the loss function of our model. The addition of this penalty discourages the model from assigning large weights to features, which helps reduce variance and improve generalization.


<img width="543" alt="image" src="https://github.com/user-attachments/assets/f1728d88-cbdb-46c0-9039-573ec8d86d10" />

### **Conclusions** 

This capstone explored how machine learning models of varying complexity perform. We started with polynomial regression, demonstrating how increasing the polynomial degree can improve performance on non-linear data, but also risks overfitting if not properly regularized, as we saw with the Polynomial Degree = 20 being corrected with the Ridge regression. 

A particularly important takeaway came from the comparison between a cimple neural network (with a single hidden layer with 50 neurons) and a deeper neural network (three hidden layers with 400, 300, and 200 neurons). While the simple model achieed decent accuracy on the MNIST dataset, the deeper model significantly outperformed it, showing higher accuracy and better generalization. This highlights how model capacity and architecture play a critical role when working with complex datasets like images, where deeper models can learn richer representations. 

Ultimately, this notebook reinforces key concepts ike underfitting, overfitting, regularization, and the importance of model selection based on data complexity. It also demonstrates the practical impact of these choices through clear visualizations and real-world performance metrics. 

### **Resources**

Youtube tutorial link: https://www.youtube.com/watch?v=j5WGoANPYnc
Google colab: https://colab.research.google.com/github/lopezbec/intro_python_notebooks/blob/master/Capstone_4_DS201.ipynb#scrollTo=s5wBSIIIKDLG
