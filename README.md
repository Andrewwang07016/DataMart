
1. Setup

We begin by setting up the title, author, and output of your programming project. This code shows that this chunk will not be included in final HTML document, and the libraries or packages that you will be using have been loaded in.


2. Getting the Data Ready

In this section we load the data into the dataframe. Then took a quick look at the data structure, and a summary of statistics. After checking for missing values, we convert the catergorical variables into factors for modeling, selecting only revelent columns for analysis.


3. Exploratory Data Analysis

The first visual shows the total purchase amounts by product category and gender. The second shows spending patterns by age and gender.



4. Feature Selection

We used leaps to find optimal predictors, this uses a search to identify which predictors or variables best fit the R² or the proportion of variance in the dependent variable.


5. Model 1: Linear Regression

In Model 1 we split the data into 80 training and 20 validation sets. Then we train a linear regression model with 10-fold cross-validation. Doing this allows us to preform analysis that uncovers R², or accuracy, RMSE, or error magnitude, and MAE, or average error.


6. Model 2: Random Forest

Speeding up the model training by using multiple CPU cores, the purpose of this model to capture non-linear relationships. Additionally we evaluate the performance of the Random Forest model on the validation data.
