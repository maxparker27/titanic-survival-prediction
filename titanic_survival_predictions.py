import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, linear_model
from enum import Enum, auto


class DataType(Enum):
    DATASET = auto()


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

    def missing_values(self) -> dict:
        """
        Obtain the number of missing values in each column.
        """

        column_missing_values = {}

        for column in self.dataset:

            column_missing_values[column] = self.dataset[column].isna(
            ).sum()

        print("Number of missing values in each column -> {}".format(column_missing_values))

        return column_missing_values

    def preparing_data(self) -> DataType.DATASET:
        """
        Preparing dataset so that the model can train on it properly.
        """

        self.dataset = self.dataset.drop(
            columns=["PassengerId", "Ticket", "Cabin", "Embarked", "SibSp", "Parch", "Name"])

        self.dataset = self.dataset.dropna().drop_duplicates()

        """
        Encoding Sex Variable:
            Female: 1
            Male: 0
        """
        for i, row in self.dataset.iterrows():
            if self.dataset["Sex"][i] == "female":
                self.dataset.at[i, "Sex"] = 1
            else:
                self.dataset.at[i, "Sex"] = 0

        return self.dataset

    def running_model(self) -> float:
        """
        Obtain accuracy of prediction model on Titanic Dataset.
        """

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

    def column_class_distribution(self, column: list) -> dict:
        """
        Obtain frequency and percentage of each occurence in a particular column of the dataset.
        """

        track_frequencies = {}
        track_percentages = {}

        for row in column:
            if row not in track_frequencies.keys():
                track_frequencies[row] = 1
            else:
                track_frequencies[row] += 1

        print("Frequency of each value: {}".format(track_frequencies))

        for key, value in track_frequencies.items():
            track_percentages[str(key) +
                              "_percentage"] = (value / len(column)) * 100

        print("Percentage of each value of total: {}".format(track_percentages))

        return track_frequencies, track_percentages

    def __str__(self):
        return(str(self.dataset))


if __name__ == "__main__":
    training_data = pd.read_csv("train.csv")

    dataset = Titanic(training_data)
    dataset.missing_values()
    dataset.preparing_data()
    dataset.column_class_distribution(training_data["Sex"])
    dataset.running_model()
