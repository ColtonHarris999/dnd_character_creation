"""
Colton Harris
CSE 163
D&D Character Creation Analysis

Develops and trains the neural network used
to predict the class of a dnd character given
that character's AC, HP, and ability scores.
Also plots the balance of the data and the
accuracy of the model accross the classes.
"""


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    """
    A feed forward nueral network desinged to predict
    D&D character classes.

    Args:
        nn (Module): base class for all nueral networks
    """
    def __init__(self) -> None:
        """
        Initializes the nueral network
        """
        super().__init__()
        # fully connected layers
        self._fc1: nn.Linear = nn.Linear(8, 32)  # input, output
        self._fc2: nn.Linear = nn.Linear(32, 32)  #
        self._fc3: nn.Linear = nn.Linear(32, 32)  #
        self._fc4: nn.Linear = nn.Linear(32, 32)  #
        self._fc5: nn.Linear = nn.Linear(32, 13)  # 13 classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The forward pass computation. This is how the network takes
        in new data, passes it through the nodes, and returns
        the end result.

        Args:
            x (torch.Tensor): The current tensor to run through the network

        Returns:
            torch.Tensor: The ouput nodes after the given data is ran through
        """
        # run activation function (rectify linera) on layers 1-3
        x = F.relu(self._fc1(x))
        x = F.relu(self._fc2(x))
        x = F.relu(self._fc3(x))
        x = F.relu(self._fc4(x))
        x = self._fc5(x)
        return F.log_softmax(x, dim=1)  # dim=1 flat linear layer


def plot_classes(char_data: gpd.GeoDataFrame) -> None:
    """
    Generates two figures, both only show characters who do not multiclass.
    1) A figure with three graphs, each showing the percent of characters
       of the core archetypes (martial, half-casters, full casters)
       Saves the file as 'class_archetypes_percent.png'
    2) A figure with a graph displaying the distribution of individual classes
       Saves the file as 'classes_percent.png'

    Args:
        char_data (gpd.GeoDataFrame): D&D character dataset
    """
    # filter down to only classes and convert back to df from gpdf
    classes = pd.DataFrame(char_data[['justClass']])

    # D&D Classes
    # print(set(classes['justClass'].values))
    martial = {'Barbarian', 'Fighter', 'Rogue', 'Monk'}
    half_casters = {'Artificer', 'Ranger', 'Paladin'}
    full_casters = {'Bard', 'Cleric', 'Druid', 'Sorcerer', 'Warlock', 'Wizard'}
    all_classes = martial | half_casters | full_casters

    # Filter by archetype
    martial_classes = classes[classes['justClass'].isin(martial)]
    half_casters_classes = classes[classes['justClass'].isin(half_casters)]
    full_casters_classes = classes[classes['justClass'].isin(full_casters)]
    all_single_classes = classes[classes['justClass'].isin(all_classes)]

    # Get counts
    martial_classes = martial_classes.groupby(
        'justClass')['justClass'].count()
    half_casters_classes = half_casters_classes.groupby(
        'justClass')['justClass'].count()
    full_casters_classes = full_casters_classes.groupby(
        'justClass')['justClass'].count()
    all_single_classes = all_single_classes.groupby(
        'justClass')['justClass'].count()

    # Get percent
    martial_classes = martial_classes \
        / sum(martial_classes.values) * 100
    half_casters_classes = half_casters_classes \
        / sum(half_casters_classes.values) * 100
    full_casters_classes = full_casters_classes \
        / sum(full_casters_classes.values) * 100
    all_single_classes = all_single_classes \
        / sum(all_single_classes.values) * 100

    # Plotting Fig 1
    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    # Martial
    martial_classes.plot(ax=ax1, kind='bar',
                         xlabel='Martial Class', ylabel='% Played',
                         color='maroon')
    ax1.set_title('D&D Martial Classes %')
    # Half Casters
    half_casters_classes.plot(ax=ax2, kind='bar',
                              xlabel='Half Caster Class', ylabel='% Played',
                              color='green')
    ax2.set_title('D&D Half Caster Classes %')
    # Full Casters
    full_casters_classes.plot(ax=ax3, kind='bar',
                              xlabel='Full Caster Class', ylabel='% Played')
    ax3.set_title('D&D Full Caster Classes %')
    fig1.suptitle('D&D Percent Class Archetypes Played')
    plt.savefig('class_archetypes_percentages.png', bbox_inches="tight")

    # Plotting Fig 2
    fig2, ax4 = plt.subplots(1, 1, figsize=(15, 5))
    all_single_classes.plot(ax=ax4, kind='bar',
                            xlabel='Class', ylabel='% Played',
                            color='silver')
    ax4.set_title('D&D Percent Classes Played')
    plt.savefig('class_percentages.png', bbox_inches="tight")


def train(train_set: list[(torch.Tensor, torch.Tensor)],
          test_set: list[(torch.Tensor, torch.Tensor)]) -> Net:
    """
    Creates, trains, and returns a new network using the given
    training and testing datasets.
    Prints the accuracy of the testing and training data as well as
    the loss for reach run through the data.

    Args:
        train_set (list[(torch.Tensor, torch.Tensor)]): training dataset
            formatted as: (features tuple, labels tuple)
        test_set (list[(torch.Tensor, torch.Tensor)]): testing dataset
            formatted as: (features tuple, labels tuple)

    Returns:
        Net: The trained network
    """
    net = Net()
    optimizer = optim.Adam(net.parameters(), lr=0.0018)
    EPOCHS = 6  # make six passes through data

    # Training loop
    for EPOCH in range(EPOCHS):
        for data in train_set:
            # data is a tuple of features and labels
            X, y = data
            net.zero_grad()
            output: torch.Tensor = net(X.view(1, -1))
            # get loss
            loss = F.cross_entropy(output, y.view(1))
            loss.backward()  # back-propegate the loss
            optimizer.step()  # optimize the model
        print(loss)

    # Find accuracy on train data:
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in train_set:
            output = net(X.view(1, -1))
            predicted = torch.argmax(output)
            if predicted == y:
                correct += 1
            total += 1
    print("Train accuracy: ", round(correct/total, 2))

    # Find accuracy on test data:
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_set:
            output = net(X.view(1, -1))
            predicted = torch.argmax(output)
            if predicted == y:
                correct += 1
            total += 1
    print("Test accuracy: ", round(correct/total, 2))
    return net


def plot_class_accuracy(net: Net, data: list[(torch.Tensor, torch.Tensor)],
                        key: dict[int, str]) -> None:
    """
    Plots the accuracy and inaccuracy of the network.

    Args:
        net (Net): The network to be analyzed
        data (list[): The data used to find accuracy and inaccuracy
        key (dict[int, str]): The dictionary used to revert the label
                              encodings back to D&D classes.
    """
    counts = {i: 0 for i in key.keys()}
    correct_counts = {i: 0 for i in key.keys()}
    incorrect_counts = {i: 0 for i in key.keys()}
    with torch.no_grad():
        for X, y in data:
            output = net(X.view(1, -1))
            predicted = torch.argmax(output)
            y_num = y.item()
            pred_num = predicted.item()
            if predicted == y:
                correct_counts[y_num] = correct_counts[y_num] + 1
            else:
                incorrect_counts[pred_num] = incorrect_counts[pred_num] + 1
            counts[y_num] = counts[y_num] + 1
    correct_percent = {
        key[i]: 100 * correct_counts[i] / counts[i] for i in key.keys()}
    incorrect_percent = {
        key[i]: 100 * incorrect_counts[i] / counts[i] for i in key.keys()}

    # Plotting Accuracies
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    percent_series = pd.Series(correct_percent)
    percent_series.plot(ax=ax1, kind='bar',
                        xlabel='Class', ylabel='Accuracy %', color='green')

    ax1.set_title('D&D Class Predictor Accuracies')
    plt.savefig('class_predictions_accuracies.png', bbox_inches="tight")

    # Plotting Inaccuracies
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 5))
    in_percent_series = pd.Series(incorrect_percent)
    in_percent_series.plot(ax=ax1, kind='bar',
                           xlabel='Class', ylabel='Inaccuracy %', color='red')

    ax1.set_title('D&D Class Predictor Inaccuracies')
    plt.savefig('class_predictions_inaccuracies.png', bbox_inches="tight")
