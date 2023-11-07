# ELM-with-Iterative-Optimizations
A PyTorch classifier and regression extreme learning machine that can utilize iterative solving methods to set hidden-layer to output weights. 

Project Start: 11/3/2023
Update 11/6/23: 
After one weekend, the model can be applied to the MNIST dataset and can produce classification accuracies up to ~97% (Moore-Penrose Pseudo-Inv and 1000 hidden neurons).
The iterative optimization methods (successive-over relaxation, Gauss-Seidel, Jacobi) seem to produce results just as quickly and accuracy is only partially dimenished.
A full analysis of training time, classification/regression accuracy, and RAM usage for the different optimization methods on more datasets is in order.

Update 11/7/2023:
Implementation of the BUPA and Iris datasets from UCI Machine Learning Repository. Now have examples of both the regressor and classifier working properly. 
In the above mentioned datasets, I need to implement a shuffling of data and a breakdown of the dataset to seperate the training and test data. Next steps are still accuracy, timing, and efficiency.
