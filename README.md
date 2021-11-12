# Boston-house-price-prediction-using-machine-Learning


Boston-House-Price-Pridiction
In this project i have used different machine learning algorithm for better prediction.

Report

Overview Importing libraries and Reaading Dataset Data Analysis Data Preprocessing Visualizing Data Creating a pipeline for feature Scaling Selecting desired Models Linear Regression Decision Tree Regression Random Forest Regression Error Measure in Models Visualizing Model Performance Final Conclusion

Overview .. _boston_dataset:
Boston house prices dataset Data Set Characteristics:

:Number of Instances: 506

:Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.

:Attribute Information (in order): - CRIM per capita crime rate by town - ZN proportion of residential land zoned for lots over 25,000 sq.ft. - INDUS proportion of non-retail business acres per town - CHAS Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) - NOX nitric oxides concentration (parts per 10 million) - RM average number of rooms per dwelling - AGE proportion of owner-occupied units built prior to 1940 - DIS weighted distances to five Boston employment centres - RAD index of accessibility to radial highways - TAX full-value property-tax rate per $10,000 - PTRATIO pupil-teacher ratio by town - B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town - LSTAT % lower status of the population - MEDV Median value of owner-occupied homes in $1000's

:Missing Attribute Values: None

:Creator: Harrison, D. and Rubinfeld, D.L.

This is a copy of UCI ML housing dataset. https://archive.ics.uci.edu/ml/machine-learning-databases/housing/

This dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.

The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978. Used in Belsley, Kuh & Welsch, 'Regression diagnostics ...', Wiley, 1980. N.B. Various transformations are used in the table on pages 244-261 of the latter.

The Boston house-price data has been used in many machine learning papers that address regression problems.

.. topic:: References

Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261. Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.

Importing Libraries and Reading the Dataset

Data Analysis

.head() and .tail() Head and tail functions are capable of 5 rows per time. But you can change this situation. So you can enter the desired value in the parameter section. The first function, ie head (), returns the initial values. The second function returns the last values.

.info() Then, the data has what informations. We are learning the information for all data The info function shows the data types and numerical values of the features in our data set. In short, this information about our data set. :)

Data Preprocessing
.isnull () Detects missing values. Return a boolean same-sized object indicating if the values are NA. NA values, such as None or numpy.NaN, gets mapped to True values. Everything else gets mapped to False values. Characters such as empty strings '' or numpy.inf are not considered NA values (unless you set pandas.options.mode.use_inf_as_na = True).

All data contained in our data set have been checked to check for any null values. So there is no problem left for a general analysis.

Checking for missing value. There is not any missing values as shown below. Now,I will check null on all data and If data has null, I will sum of null data's. In this way, how many missing data is in the data.

.describe() After I get the main intuition, I am investigating further to see some analytical attributes.

Describe function includes analysis of all our numerical data. For this, count, mean, std, min,% 25,% 50,% 75, max values are given. The reason this section is important is that you can estimate the probability that the values found here are deviant data.

.hist() It plots histogram with customizable features.

A histogram is a graphical display of data using bars of different heights. In a histogram, each bar groups numbers into ranges. Taller bars show that more data falls in that range. A histogram displays the shape and spread of continuous sample data.

Everything fine up to now, Yet i will create histogram for better insights.

Dataframe.iloc[] Here we are seprating X and Y columns for Test and prediction

Pandas provide a unique method to retrieve rows from a Data frame. Dataframe.iloc[] method is used when the index label of a data frame is something other than numeric series of 0, 1, 2, 3….n or in case the user doesn’t know the index label. Rows can be extracted using an imaginary index position which isn’t visible in the data frame.

Splitting the dataset The train_test_split function is for splitting a single dataset for two different purposes: training and testing. The testing subset is for building your model. The testing subset is for using the model on unknown data to evaluate the performance of the model.

Stratified Sampling Stratified Sampling: This is a sampling technique that is best used when a statistical population can easily be broken down into distinctive sub-groups. Then samples are taken from each sub-groups based on the ratio of the sub groups size to the total population.

Now I want to check whether the Stratified Sampling is working or not. So i have chosen a Feature in which Data is inputted in 0 and 1. So i will be checking that the distribution of zeroes and ones are same in Train data and Test data or not.

So Again i will be checking the Features and their statistical interpretation.

Now we are good to go, let's see the variety of datatypes in our dataset

Visualizing Data
.corr() Pandas dataframe.corr() is used to find the pairwise correlation of all columns in the dataframe. Any na values are automatically excluded. For any non-numeric data type columns in the dataframe it is ignored.

So here i will be looking for correlation in Features.

.heatmap() For better insight i will plot heatmap.

The Big colorful picture below which is called Heatmap helps us to understand how features are correlated to each other. Postive sign implies postive correlation between two features whereas Negative sign implies negative correlation between two features.

One Visualization to Rule Them All We will perform analysis on the training data. The relationship between the features found in the training data is observed. In this way, inference about the properties can be made.

sns.pairplot Seaborn Pairplot uses to get the relation between each and every variable present in Pandas DataFrame. It works like a seaborn scatter plot but it plot only two variables plot and sns paiplot plot the pairwise plot of multiple features/variable in a grid format.

Creating a pipeline for feature scaling SimpleImputer is a scikit-learn class which is helpful in handling the missing data in the predictive model dataset. It replaces the NaN values with a specified placeholder. It is implemented by the use of the SimpleImputer() method which takes the following arguments :
missing_values : The missing_values placeholder which has to be imputed. By default is NaN stategy : The data which will replace the NaN values from the dataset. The strategy argument can take the values – ‘mean'(default), ‘median’, ‘most_frequent’ and ‘constant’. fill_value : The constant value to be given to the NaN data using the constant strategy.

Standardization scales each input variable separately by subtracting the mean (called centering) and dividing by the standard deviation to shift the distribution to have a mean of zero and a standard deviation of one.

sklearn provides a class called StandardScaler which will standerdise the data.

Pipeline A machine learning pipeline is used to help automate machine learning workflows. They operate by enabling a sequence of data to be transformed and correlated together in a model that can be tested and evaluated to achieve an outcome, whether positive or negative.

Selecting desired models
Linear Regression In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). The case of one explanatory variable is called simple linear regression. For more than one explanatory variable, the process is called multiple linear regression. This term is distinct from multivariate linear regression, where multiple correlated dependent variables are predicted, rather than a single scalar variable.

Decision Tree Regression Decision Tree - Regression. Decision tree builds regression or classification models in the form of a tree structure. It breaks down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes.

Random Forest Regression A random regression forest is an ensemble of randomized regression trees. Denote the predicted value at point by the -th tree, where are independent random variables, distributed as a generic random variable , independent of the sample .

8.Error Measure in Models Here we will be measuring the Error in the Respective Models.

Visualizing Model Performance sns.barplot A bar plot represents an estimate of central tendency for a numeric variable with the height of each rectangle and provides some indication of the uncertainty around that estimate using error bars.
10.Final Conclusion So In this notebook kernel, I have deployed three regression models using the Boston Housing Dataset. These are linear regression,decision tree regression and random forest regression. Afterward I have visualized and calculated the performance measure of the models. Out of which Random forest regression is the best suit for this dataset.

Please make a valuable comment and let me know if any ,how to improve the performance of the model, visualization, preprocessing, Analysis or something in this kernel. This will definitely help me in future.
