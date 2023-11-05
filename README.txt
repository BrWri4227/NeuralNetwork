
Brycen Wright
Date: April 13th, 2023
Course: CIS 3700
Professor: K. Moylan
Assignment: Assignment 4
Due Date: April 14th, 2023

To run this program:
Download main.py, add your data file in CSV format to the same directory as the main. Go into main.py and adjust the input_size variable to the proper value (784 for MNIST) and then run by typing:
    py main.py

By default, it has 784 input nodes, 1 hidden layer containing 100 nodes.
These can be changed by modifying the following variables:
    hiddenLayerCount
    hiddenLayerSize
Adjust them to any positive integer values you please.


Analysis:

The various training and testing networks were all trained and testing using 100 hidden nodes and 10 epochs.
Based purely off of the numbers given from the network, the best architecture would be one with two hidden layers, as it provided the highest accuracies across all 10 epochs.
Throughout the testing and training process, it seemed that too few hidden nodes and the network performed not very well on both the training and testing set most likely due to a lack of convergenece resulting in a over-generalized network.
On the flip side, with 3 hidden layers, the network performed significantly worse, performing the worst on both the train and testing set most likely due to an overfitting of the data.
With 2 hidden layers, the network performed much better attaining the highest accuracies on both the training and testing set.
This is most likely due to the structure of the MNIST dataset/problem that 2 hidden layers worked out the best.
Maybe if we tested with different numbers of hidden nodes as well we could find an even more optimal solution.
In conclusion, for the MNIST data set using 2 hidden layers with 100 nodes results in the best accuracy.


Validation Accuracies (From 0 to 10 Epochs.):

0 Hidden Layers :
88.12%
89.37%
89.99%
90.34%
90.54%
90.64%
90.76%
90.85%
90.89%
91.00%

1 Hidden Layer:
88.23%
89.44%
89.97%
90.23%
90.52%
90.69%
90.83%
90.96%
91.02%
91.11%

2 Hidden Layer:
88.13%
89.30%
89.99%
90.57%
90.89%
91.23%
91.48%
91.59%
91.65%
91.72%

3 Hidden Layers:
87.40%
89.04%
89.48%
89.42%
89.41%
89.46%
89.33%
89.61%
89.78%
89.95%
