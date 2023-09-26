# Logistic Regression

<p align="center">
  <!-- Insert an appropriate image or graphic representation of Logistic Regression, if you have one -->
</p>

---

## Description

Logistic regression is a popular statistical method used for modeling the probability of a certain 
class or event and is rudimentary for anyone getting started with machine learning. The goal of Logistic
Regression is: given an input x, predict the probability that it belongs to the class y.

To accomplish this, we obtain a dataset containing inputs and labels. Next, we preprocess the dataset
in order to train and test the logistic regression model (e.g. vectorizing, reshaping, normalization, etc.). 
After, we initialize our model; setting up the weights and bias so that we can peform a linear transform on
vectorized inputs. We then define an activation function (sigmoid) to squeeze the linear transform into 
the interval [0, 1] (the resultant value tells us the liklihood that x is y). Following this, we define 
our loss function (log loss for binary classification) which we can use to tell us the cost of our weights
on the function (i.e. how good or bad the current state of the model is at predicting an input x to its
respective label y). Using a training loop, we can iteratively apply the forward pass of the model to the
inputs to get our model's predictions. We'll calculate the loss and use a little bit of calculus to find
the gradients (i.e. the negative slope of a function at a point) at the weights and bias respectively. 
To "learn" our weights and bias, we can apply a small constant(learning rate) to their gradients and subract
them from the current weights and bias respecitvely. Hopefully after many iterations, we'll have a model
that accuractely predicts inputs to their correct labels even for a dataset the model hasn't seen.

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
