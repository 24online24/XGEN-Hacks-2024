from enum import Enum


class MODEL_NAME(Enum):
    RANDOM_FOREST = "Random Forest"
    DECISION_TREE = "Decision Tree"
    LOGISTIC_REGRESSION = "Logistic Regression"
    MULTINOMIAL_NAIVE_BAYES = "Multinomial Naive Bayes"
    SUPPORT_VECTOR_MACHINE = "Support Vector Machine"


MODEL_DESCRIPTION = {
    MODEL_NAME.RANDOM_FOREST: "Random Forest is an ensemble learning method that builds multiple decision trees and outputs their majority vote for classification or average prediction for regression, enhancing accuracy and robustness.",
    MODEL_NAME.DECISION_TREE: "A Decision Tree is a model that splits data into branches based on feature values, leading to a prediction at each leaf node.",
    MODEL_NAME.LOGISTIC_REGRESSION: "Logistic Regression is a statistical model that predicts the probability of a binary outcome using a linear combination of input features.",
    MODEL_NAME.MULTINOMIAL_NAIVE_BAYES: "Multinomial Naive Bayes is a probabilistic classifier that uses Bayes' theorem to predict categories based on the frequency of features in the input data.",
    MODEL_NAME.SUPPORT_VECTOR_MACHINE: "Support Vector Machine (SVM) is a supervised learning model that finds the optimal hyperplane to separate data points into distinct classes."
}

MODEL_ACCURACY = {
    MODEL_NAME.RANDOM_FOREST: "97.57",
    MODEL_NAME.DECISION_TREE: "99.62",
    MODEL_NAME.LOGISTIC_REGRESSION: "98.79",
    MODEL_NAME.MULTINOMIAL_NAIVE_BAYES: "93.83",
    MODEL_NAME.SUPPORT_VECTOR_MACHINE: "95.74"
}
