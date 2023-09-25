# Logistic Regression

<p align="center">
  <!-- Insert an appropriate image or graphic representation of Logistic Regression, if you have one -->
</p>

---

## Description

Logistic regression is a popular statistical method used for modeling the probability of a certain 
class or event and is rudimentary for anyone getting started with machine learning.
The goal of Logistic Regression is: given an input x, predict the probability that it belongs to the class y.
To accomplish this, we obtain a dataset containing inputs and labels. Next, we preprocess the dataset
in order to train and test the logistic regression model. After, we initialize our model; setting up
the weight matrix and bias to enable the forward pass (linear transformation) of the model. 
We then define an activation function (sigmoid) to squeeze linear transformation into the interval [0, 1]
(the resultant value tells us the liklihood that x is y). Following we define our loss function
(log loss for binary classification) which tells us how right or wrong the current state of model is at
predicting inputs to their respective target class (i.e. the cost of our weights on our loss function). 
Using the loss function, we can...

> **Note**: _To stabilize numerical values for applying the sigmoid activation and calculating loss,
the input vectors to the functions are clipped within a range._

---

## Installation

1. Open your terminal and navigate to your desired directory using the command below:
   
    ```
    cd path/to/directory
    ```

2. Clone the repository using the following command:
   
    ```bash
    git clone https://github.com/Andrew011002/Logistic-Regression
    ```

---

## Dependencies

To install the necessary dependencies for the Logistic Regression model, use the command below:
  ```bash
  pip install -r dependencies.txt
  ```

---

## Notebooks

- **Cats vs Dogs Classifier**: This notebook guides you through data preprocessing and preparation, model initialization, setting up hyperparameters, and the training and testing phases. You'll witness the model achieving an accuracy rate of 99%. Access the notebook here: [cats-vs-dogs-classifier.ipynb](link-to-your-notebook).

---

## Contributing

- **Issues**: Open issues to document enhancements, bugs, or other topics relevant to this repository. Refer to the [Issues page](<your-repo-link>/issues) to begin.

- **Pull Requests**: Contribute by implementing your own improvements. This can range from bug fixes to more substantial changes. Link your pull requests to related open issues when possible. Navigate to the [Pull Requests page](<your-repo-link>/pulls) to start.

---

## Social

Stay connected and updated with my latest projects:

- [LinkedIn](https://www.linkedin.com/in/andrewmicholmes/)
- [Medium](https://medium.com/@andmholm)

---

## External Links

- [Huggingface Datasets API](https://huggingface.co/datasets)
- [Sigmoid Function Explanation](link-to-a-reliable-source-about-sigmoid)
- [Binary Cross Entropy Loss](link-to-a-reliable-source-about-bceloss)
