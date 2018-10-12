# r-predictiveModelingUtilities
Functions to help in model building and evaluation

`DefineFunction_ModelBuilding_Bagging.R` defines a custom function for tree-based bagging ensembles (bootstrap aggregating). There are multiple R packages that implement bagging. This script offers a couple unique features: 
* Ensembling tree models through bagging works well because tree-based models are weak learners that produce noticeably different trees when the sample is changed. Basically, they operate best with a set of diverse input trees. To further diversify the input trees, this function allows ensembling trees built with a mix of split criteria in tree building: the user can specify whether the trees' split criterion should be Gini, or Information Criterion, or a mix of both. 
* Existing R bagging functions do not give easy access to evaluating out-of-bag performance. When working with small data sets where partitioning your data or using cross-validation are not an option, evaluating out-of-bag performance can be the best available estimate of how the ensembled model would perform on new data. Note that estimates based on out-of-bag performance will be pessimistic. 

`DefineFunction_ModelEvaluation_adjustConfMatrixForPriors.R` defines a function to adjust classification model's evaluation statistics for priors, when the data used to train a model has been artificially balanced. 

`DefineFunction_ModelEvaluation_getRegMetrics.R` takes a regression-type caret _train_ object and a new data set to score, and returns select evaluation metrics. This is purely for convenience. 
