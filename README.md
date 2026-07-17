# Project-Home-Credit-Default-Risk
Data analysis project I: Home Credit Default Risk. Works on the data sets from kaggle.com/competitions/home-credit-default-risk/data.

The project is structured as follows: EDA_early_modeling dives into exploratory data analysis and looks at the results of modeling with simple Imputing/encoding and simple model choices.

Then, with the results from this notebook I develope model_selection_improv in which EDA is skipped altogether and the notebook is centered around how to improve the two initial models namely: A logistic regression and a tree-based model (initially a Random Forest). The chosen strategy for the logistic case was a WoE transforamtion using optbinning and for the tree-based method we drop random forests in favor of boosting using XGBoost. In both cases I find a significant improvement as measured by AUC for both methods and dramatically helped by feature engineering and auxiliary csv's found in the same link provided earlier.

Finally, a condensed version of model_selection_improv can be found in main.py in which the initial and final models are compared for both kind of methods and we plot several diagnostic figures of both the improvement and efficiency of the methods, as well as the relative importance of featues.

In summary we find:

 - There isn't a large difference (AUC-wise) between the boosted tree method and the WoE transformed logistic regression, thus making the regression preferable for the probabilities score produced, although the tree-based boost method is of great use do to its feature-importance rating. However, this is also provided by the WoE transformation, making the logistic regression overall preferable.

 - As found by the WoE transformation the most important features are overwhelmingly the external credit scores (EXT_SOURCE 1,2 and 3) followed by days of the credit and the overall credit term. Other features of importance include the age of the customer and the occupation of said customer but overall the credit-related information and the external credit score drive the decision making of the models (on both methods).

 - Furthermore, out of the more than 300 features considered it was found that removing up to the 50 less important features not only does not harm the model but can even marginally increase the AUC score. Further removal of features did slightly harm the logistic model.

 - Directions for further improvement include the addition of more features from the auxiliary csv's provided and the defining of more domain knowledge features obtained by simple (but well instructed) computations on the already-present features. Allocation of more computational resources could potentially improve the methods but I doubt it will be a major drive.
