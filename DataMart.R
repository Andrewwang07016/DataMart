
---
author: "Andrew Wang, Isabelle Bolen, Ahmed Elsayed, Alex Hensgen"
output: html_document
---



```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#Load in Libraries
```{r}
library(tidyverse)
library(caret)
library(ggthemes)
library(leaps)
library(doParallel)
```

## Load in Walmart Data

```{r}
Walmart <- read_csv("Data\\walmart.csv")

```

## Examine Walmart Data

```{r}
glimpse(Walmart)
summary(Walmart)
```

## Check for Missing Data

```{r}
colSums(is.na(Walmart))
```

## Convert DataTypes to Factors

```{r}
Walmart <- Walmart |> 
  mutate(Gender = as.factor(Gender),
         Age = as.factor(Age),
         Occupation = as.factor(Occupation),
         City_Category = as.factor(City_Category),
         Stay_In_Current_City_Years = as.factor(Stay_In_Current_City_Years),
         Marital_Status = as.factor(Marital_Status),
         Product_Category = as.factor(Product_Category)) |> 
  select(Purchase,Age,Gender,City_Category,Stay_In_Current_City_Years,Occupation,Marital_Status,Product_Category)
```

## Exploratory Visual 1 - Distribution of Purchase Amounts
```{r}
## Aggregate Totals by Product Category and Gender
walmart_purchase_category_gender <- Walmart |> 
  group_by(Product_Category,Gender) |> 
  summarize(Total_Purchase = sum(Purchase,na.rm=TRUE)) |> 
  arrange(desc(Total_Purchase))

## Generate Visualization
walmart_purchase_category_gender |> 
  ggplot(aes(x=Product_Category,y=Total_Purchase))+
  geom_col(aes(fill=Gender))+
  theme_minimal()+
  scale_fill_brewer(palette="Dark2")+
  labs(title = "Total Purchase by Product Category and Gender",
       x = "Product Category",
       y = "Total Purchase Amount")
```
## Exploratory Visual 2 - Purchase Amounts by Age Group and Gender
```{r}
walmart_purchase_age_groups <- Walmart |> 
  group_by(Age,Gender) |> 
  summarize(Total_purchase = sum(Purchase))
## Generate Visualization
walmart_purchase_age_groups |> 
  ggplot(aes(Age, Total_purchase))+
  geom_col(aes(fill=Gender))+
  scale_fill_brewer(palette="Dark2")+
  theme_minimal()+
  labs(title="Total Purchase Amount by Age Group and Gender",
       x= "Age Group",
       y= "Total Purchase Amount")
```


## Use Leaps Package to Determine Optimal Predictors for Linear Regression

```{r}
best_subsets <- regsubsets(Purchase~.,data=Walmart,method="exhaustive",really.big=T)
lm_summary <- summary(best_subsets)
lm_index <- which.max(lm_summary$adjr2)
best_model_adjr2 <- lm_summary$adjr2[lm_index]
best_model_adjr2

```

## Divide Training/Testing Set

```{r}
set.seed(1)
train.index <- createDataPartition(Walmart$Purchase,p=0.8,list=FALSE)
train.df <- Walmart[train.index,]
valid.df <- Walmart[-train.index,]
rm(train.index)
```

## Run Model 1 Linear Regression

```{r}
linearRegressionModel1 <- lm(Purchase ~ ., data = train.df)

#Cross Validation 10x
crossValidationModel1 <- train(Purchase ~ ., data = train.df, method = "lm", trControl = trainControl(method = "cv", number = 10))
crossValidationModel1
```

# Run on Validation Set
```{r}
test_results <- predict(crossValidationModel1,newdata=valid.df)
actuals <- valid.df$Purchase
valid_r2 <- cor(test_results,actuals)^2
valid_r2
valid_rmse <- sqrt(mean((test_results - actuals)^2))
valid_rmse
valid_mae <- mean(abs(test_results - actuals))
valid_mae
```

# Model 2

## Set up parallel processing
```{r}
cl <- makePSOCKcluster(detectCores()-1)
registerDoParallel(cl)

```


## Random Forest
```{r}
#CrossValidation setup
control <- trainControl(method="cv",number =10)

#Parameters for RandomForest Model
tuneGrid <- expand.grid(
  mtry=27,
  splitrule="extratrees",
  min.node.size=5
)

#Training RF model
set.seed(1)
randomForestModel <- train(
  Purchase~.,
  data=train.df,
  method="ranger",
  trControl=control,
  tuneGrid=tuneGrid,
  num.trees = 200
  )
print(randomForestModel)

stopCluster(cl)

```

## Validate Random Forest on Valid Set
```{r}
rf_test_results <- predict(randomForestModel,newdata = valid.df)
rf_actuals <- valid.df$Purchase
rf_valid_r2 <- cor(rf_test_results,rf_actuals)^2
rf_valid_r2
```