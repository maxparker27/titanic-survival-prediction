import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, linear_model


class Titanic:

    def __init__(self, titanic_dataset):
        self.dataset = titanic_dataset
        self.X = pd.DataFrame(dtype="object")
        self.y = pd.Series(dtype="object")
        self.X_train: list = []
        self.X_test: list = []
        self.y_train: list = []
        self.y_test: list = []
        self.model = None
        self.predictions: list = []
        self.accuracy: float = 0

    def preparing_data(self):

        self.dataset = self.dataset.drop(
            columns=["PassengerId", "Ticket", "Cabin", "Embarked", "SibSp", "Parch", "Name"])

        self.dataset = self.dataset.dropna().drop_duplicates()

        """
        Encoding Sex Variable:

            Female -> 1
            Male -> 0
        """
        for i, row in self.dataset.iterrows():
            if self.dataset["Sex"][i] == "female":
                self.dataset.at[i, "Sex"] = 1
            else:
                self.dataset.at[i, "Sex"] = 0

        return self.dataset

    def running_model(self) -> float:
        self.X = self.dataset.iloc[:, 1:]
        self.y = self.dataset.loc[:, "Survived"]

        self.X_train, self.X_test, self.y_train, self.y_test = model_selection.train_test_split(
            self.X, self.y)

        self.model = linear_model.LogisticRegression().fit(self.X_train, self.y_train)

        self.y_test = self.y_test.tolist()
        self.predictions = self.model.predict(self.X_test)

        counter: int = 0

        for i in range(len(self.X_test)):
            if self.predictions[i] == self.y_test[i]:
                counter += 1

        self.accuracy = (counter / len(self.X_test)) * 100

        print("The accuracy of the Logistic Regression model for the Titanic Dataset is {}%.".format(
            round(self.accuracy, 2)))

        return self.accuracy

    def __str__(self):
        return(str(self.dataset))


if __name__ == "__main__":
    training_data = pd.read_csv("train.csv")

    dataset = Titanic(training_data)
    dataset.preparing_data()
    dataset.running_model()
