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

Update 11/8/2023:
Implementation of Wine, Abalone, and Glass datasets. Proper batch manipulation with shuffling now included alongside time and RAM measurements. 
Time for data visualization, runs with different parameters, and maybe some fun applications such as Statcast data.

Update 11/27/2023:
Been messing around with parameters and a couple MLB classifiers for both ball in play results and pitch classification. Pitch classification has a lot of work to be done to make it a feasible model (if even possible under ELM frame), but ball in play is shockingly accurate at 85% with training costs shockingly low (less than 1mb RAM usage and less than 0.5sec training time). Better performance than generic models such as those proposed by Tyler James Burch in his preliminary blog (https://tylerjamesburch.com/blog/baseball/hit-classifier-1). 

12/11/2023:
Initial write up is complete, last update involves a hard-coded power iteration pseudo-inverse solver. Drastically inefficient compared to the built-in torch pinverse(), but this was purposeful as to compare my non-optimized iterative methods against it._
