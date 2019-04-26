# Import libraries
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os


def relationship_with_target(data, feature, path):

    """
            Plots a boxplot of the target variable against the feature passed as argument.
            :param: the dataset
            :param: feature as a string
            :param: path to the directory where you want to save the plot
            :return: the plot

    """

    plt.savefig(os.path.join(path, 'reports', 'figures', 'relationship_of_target_with_{}.png'.format(feature)))

    return sns.boxplot(x=data[feature], y=data['Purchase'])
