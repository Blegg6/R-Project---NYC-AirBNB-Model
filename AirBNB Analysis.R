---
  title: "Airline Neighborhood"
author: "Billy Legg"
date: "2024-04-24"

---
  
  ```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

#load data
nyc<- read.csv("BMSO603 Files/Final Project/NYC_ABNB.csv")
nyc<- nyc[!is.na(nyc$price),]
nyc$price<- na.omit(nyc$price)
nyc$room_type<- na.omit(nyc$room_type)
nyc$minimum_nights[is.na(nyc$minimum_nights)]<- 0
nyc$reviews_per_month[is.na(nyc$reviews_per_month)]<- mean(nyc$reviews_per_month,na.rm=TRUE)



#Remove unnecessary columns
nyc$id<- NULL
nyc$host_id<- NULL


nyc$room_type<- factor(nyc$room_type, levels = c("Shared room", "Private room", "Entire home/apt"))

#Create Brooklyn/Manhattan column
add_brooklyn_manhattan_column <- function(data) {
  data$Brooklyn_Manhattan <- ifelse(data$neighbourhood_group %in% c("Brooklyn", "Manhattan"), "Yes", "No")
  return(data)
}

nyc <- add_brooklyn_manhattan_column(nyc)

nyc$Brooklyn_Manhattan<- factor(nyc$Brooklyn_Manhattan, levels = c("No","Yes"))

#basic exploratory analysis

#bar graph - neighbourhood
average_price_by_neighbourhood <- aggregate(price ~ neighbourhood_group, data = nyc, FUN = mean)

library(ggplot2)
```{r pressure, echo=FALSE}
ggplot(average_price_by_neighbourhood, aes(x = neighbourhood_group, y = price)) +
  geom_col(fill = "skyblue", color = "black") +
  labs(title = "Average Price by Neighbourhood Group",
       x = "Neighbourhood Group",
       y = "Average Price")
```

#Bar graph - room
average_price_by_room <- aggregate(price ~ room_type, data = nyc, FUN = mean)

```{r pressure, echo=FALSE}
ggplot(average_price_by_room, aes(x = room_type, y = price)) +
  geom_col(fill = "skyblue", color = "black") +
  labs(title = "Average Price by Room Type",
       x = "Room Type",
       y = "Average Price")
```

```{r}
#ANOVA neighbourhood
ANOVA_neighborhood<- aov(price ~ neighbourhood_group, data = nyc)
summary(ANOVA_neighborhood)
TukeyHSD(ANOVA_neighborhood)
```
```{r}
#ANOVA room type
ANOVA_room<- aov(price ~ room_type, data = nyc)
summary(ANOVA_room)
TukeyHSD(ANOVA_room)
```

#Partition Data
set.seed(1234)
datatrain<- sample(nrow(nyc),.6*nrow(nyc))
nyctrain<- data.frame(nyc[datatrain,])
nyctest<-data.frame(nyc[-datatrain,])

```{r}
nyclm2<- lm(nyctrain$price ~ room_type + number_of_reviews + availability_365 + neighbourhood_group, data = nyctrain)
summary(nyclm2)
```

nyc_train_residuals2<- nyclm2$residuals

```{r}
(MSE_lm_train2<- (mean(nyc_train_residuals2^2)))
(RMSE_lm_train2<- sqrt(mean(nyc_train_residuals2^2)))
```

```{r}
nyctrain$averageprice<- mean(nyctrain$price)
nyctrain$baselineresidual<- nyctrain$price-nyctrain$averageprice
(MSE_baseline_train<- (mean(nyctrain$baselineresidual^2)))
(RMSE_baseline_train<- sqrt(mean(nyctrain$baselineresidual^2)))
```
#RMSE of the linear model outperforms RMSE of the baseline (231.20 vs 241.1456)

#Repeat steps for test RMSE 
validation_predicted_values2<- predict(nyclm2, newdata = nyctest)
validation_prediction_errors2<- nyctest$price - validation_predicted_values2

```{r}
(MSE_test2<- (mean(validation_prediction_errors2^2)))
(RMSE_test2<- sqrt(mean(validation_prediction_errors2^2)))
```

nyctest$averageprice<- mean(nyctest$price)
nyctest$baselineresidual<- nyctest$price-nyctest$averageprice

```{r}
(MSE_baseline_test<- (mean(nyctest$baselineresidual^2)))
(RMSE_baseline_test<- sqrt(mean(nyctest$baselineresidual^2)))
```
#test residuals are 228.54 and 238.65 for baselines


#Now random tree
library(randomForest)

#bagging


set.seed(1)
bag.nyc2=randomForest(price~ neighbourhood_group + room_type + minimum_nights + number_of_reviews + reviews_per_month + calculated_host_listings_count + availability_365,data=nyctrain,mtry=7,importance=TRUE)
```{r}
bag.nyc2
```

``` {r}
importance(bag.nyc2)
```
```{r}
varImpPlot(bag.nyc2)
```

sqrt(54147.24)

```{r}
yhat.bag = predict(bag.nyc2,newdata=nyctest)
```

```{r}
#RMSE_bag
sqrt(mean((yhat.bag-nyctest$price)^2))
```
```{r}
#MSE Bag
(mean((yhat.bag-nyctest$price)^2))
```

#random forest
library(randomForest)


#Random forest 2 - started with 4
set.seed(1)
rf.nyc2=randomForest(price~ neighbourhood_group + room_type + minimum_nights + number_of_reviews + reviews_per_month + calculated_host_listings_count + availability_365,data=nyctrain,mtry=2,importance=TRUE)

```{r}
rf.nyc2
```
```{r}
importance(rf.nyc2)
```

```{r}
varImpPlot(rf.nyc2)
```
#RMSE of random forest training
sqrt(48633.94)

yhat.rf2 = predict(rf.nyc2,newdata= nyctest)

```{r}
#RMSE of RF test
sqrt(mean((yhat.rf2-nyctest$price)^2))
```

```{r}
#MSE of RF
(mean((yhat.rf2-nyctest$price)^2))
```



#KNN Test
knn_nyc<- nyc
knn_nyc$name<- NULL
knn_nyc$host_name<- NULL
knn_nyc$neighbourhood_group<- NULL
knn_nyc$neighbourhood<- NULL
knn_nyc$latitude<- NULL
knn_nyc$longitude<- NULL
knn_nyc$room_type<- NULL
knn_nyc$last_review<- NULL
knn_nyc$Brooklyn_Manhattan<- NULL


#Partition KNN data
set.seed(1234)
inTrainknn <- sample(nrow(knn_nyc), 0.6*nrow(knn_nyc))
#
knntrain <- knn_nyc[inTrainknn,] #with 60% of the data
dftempknn <- knn_nyc[-inTrainknn,] # with 40% of the data
##
# Now split the dftemp data set in two equal parts
inValknn <- sample(nrow(dftempknn), 0.5*(nrow(dftempknn)))
# 
knnvalidation <- dftempknn[inValknn,]
knntest<- dftempknn[-inValknn,]
dftempknn <- NULL


#used the below formula to get the optimum k of 146
library(kknn)
price.kknn <- kknn(price~., knntrain, knnvalidation, distance = 2, k = 146,
                   kernel = "rectangular", scale = T)


```{r}
plot(price.kknn$fitted.values, knnvalidation$price, xlab = "Fitted Values of Price", ylab="Actual Price")
abline(0,1)
#
```

#Get KNN Fitted values
price.kknn$fitted.values

```{r}
(MSE_KNN <- (mean((knnvalidation$price-price.kknn$fitted.values)^2)))
RMSE_knn = sqrt(mean((knnvalidation$price-price.kknn$fitted.values)^2))
RMSE_knn

RMSE_knn_test = sqrt(mean((knntest$price-test_knn_predict)^2))
RMSE_knn_test

(MSE_KNN_Test <- (mean((knntest$price-test_knn_predict)^2)))
```


kmax <- 150
RMSE_knn1 <- rep(0,kmax)

#Find optimal K to be plugged in above.
for (i in 1:kmax){
  set.seed(1234)
  kknn <- kknn(price~., knntrain, knnvalidation, distance = 2, k = i,
               kernel = "rectangular", scale = T)
  
  RMSE_knn1[i] <- sqrt(mean((knnvalidation$price-kknn$fitted.values)^2))
}

#gives us the value of K to plug in above, in this case it is 146
```{r}
which.min(RMSE_knn1)
```