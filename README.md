# 🌼 Iris Flower Classifier - Gradio App

This is a **Gradio-based interactive web app** that predicts the species of an Iris flower based on its measurements. The app uses a **pre-trained TensorFlow/Keras model** to classify the flower as:

- Iris-setosa  
- Iris-versicolor  
- Iris-virginica  

---

## Features

- Manual input of **sepal length, sepal width, petal length, and petal width**.  
- Validation of inputs based on dataset ranges:  
  - Sepal Length: 4.3 – 7.9 cm  
  - Sepal Width: 2.0 – 4.4 cm  
  - Petal Length: 1.0 – 6.9 cm  
  - Petal Width: 0.1 – 2.5 cm  
- Displays **predicted species** along with **confidence percentage**.  
- Built with **Gradio** for an easy-to-use web interface.  

---




## Requirements

- Python 3.8+  
- TensorFlow  
- scikit-learn  
- numpy  
- pandas  
- Gradio  

You can install dependencies via:

```bash
pip install tensorflow scikit-learn numpy pandas gradio
