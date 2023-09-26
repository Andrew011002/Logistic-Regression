# Logistic Regression

<p align="center">
  <img src="https://github.com/Andrew011002/Logistic-Regression/blob/master/logistic-regression-model-3.png" alt="Logistic Regression Model">
</p>

---

## Description

Logistic regression[^1] is a rudimentary statistical method commonly employed for modeling the probability of a given class or event.
It serves as a fundamental starting point for anyone venturing into machine learning. The objective is to predict the likelihood
that a given input _x_ belongs to a specific class _y_.  

To achieve this, a dataset containing relevant inputs and labels is first obtained. The dataset is then preprocessed, which
involves steps like vectorizing, reshaping, and normalizing, to make it suitable for training and testing the logistic
regression model. The model itself is initialized by setting up the weights and bias, which will be used for a linear
transformation of the preprocessed inputs. Following this, a sigmoid[^2] activation function is applied to squeeze the linearly
transformed inputs into an interval [0, 1]. This result provides an estimate of the probability that _x_ is _y_. To assess the
model's performance, a loss function—specifically log-loss[^3] for binary classification—is utilized. This gives an indication of how
closely the model's predicted probabilities align with the actual labels. With the model, loss function, and data in place, a
training loop is executed. During each iteration of this loop, a forward pass is performed to generate the model's predictions,
after which the loss is calculated. Calculus is then used to find the gradients with respect to the weights and bias. These
gradients, when scaled by a learning rate, are used to update the model's weights and bias term. Ideally, after numerous
iterations, the model should be capable of making accurate predictions, even on unseen data.

> **Note**: _To stabilize numerical values for applying the sigmoid activation and calculating loss
for this implementation of logistic regression, the input vectors to the functions are clipped within a range._

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

## Scripts

- **Logistic Regression script**: This python file shows you the inner-workings of a logistic regression model implemented in the class `LogisticRegression`. You can not only inspect the model, but see some of the other functions used to enable it's capabilities like: `flatten()` for data preprocessing, `sigmoid()` for the activation function, `calculuate_loss()` for the log-loss function, etc. You can access the full python script here: [logistic_regression.py](https://github.com/Andrew011002/Logistic-Regression/blob/master/logistic_regression.py).

## Notebook

- **Cats vs Dogs Classifier**: This notebook guides you through building a `LogisticRegression` model to predict cats and dogs from the [cats_vs_dogs Hugging Face Dataset](https://huggingface.co/datasets/cats_vs_dogs). You'll learn basic data preprocessing, model initialization, setting up hyperparameters, and both the training and testing of the model. You should expect to see the model achieving a 99% when the notebook is run successfully. You can access the notebook here: [cats-vs-dogs-classifier.ipynb](https://github.com/Andrew011002/Logistic-Regression/blob/master/cats_vs_dogs.ipynb).

---

## Contributing

- **Issues**: Open issues to document enhancements, bugs, or other topics relevant to this repository. Refer to the [Issues page](https://github.com/Andrew011002/Logistic-Regression/issues) to begin.

- **Pull Requests**: Contribute by implementing your own improvements. This can range from bug fixes to more substantial changes. Link your pull requests to related open issues when possible. Navigate to the [Pull Requests page](https://github.com/Andrew011002/Logistic-Regression/pulls) to start.

---

## Social

Stay connected and updated with my latest projects:

- [LinkedIn](https://www.linkedin.com/in/andrewmicholmes/)
- [Medium](https://medium.com/@andmholm)

---

## External Links

[^1]: [Logistic Regression](https://www.ibm.com/topics/logistic-regression)
[^2]: [Sigmoid Function Explanation](https://en.wikipedia.org/wiki/Sigmoid_function)
[^3]: [Binary Cross Entropy Loss](https://arize.com/blog-course/binary-cross-entropy-log-loss)
