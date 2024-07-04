Objective: To analyze the factors that influence the pricing of Airbnb listings, including location (neighbourhood_group, neighbourhood), room type, and availability.
Hypothesis: Listings in some areas are more expensive than other boroughs, and entire homes/apartments command higher prices than private or shared rooms.

Original data source can be accessed at: http://insideairbnb.com/new-york-city

	Our solution for analyzing the factors that influence pricing is twofold. We will create various models to test which is most appropriate for predicting price in a statistically significant manner.  Likewise, we will determine if the mean price rentals have significant variances across different neighborhoods and room types.  These analyses will allow us to answer the primary questions of our objectives: “What factors significantly influence price?” and “Are there significant differences in the prices of rentals in certain boroughs or in certain room types?”
	To start, a simple ANOVA test can give us some information about the price variances across the multiple neighborhoods and room types.  Once a simple one-way ANOVA test is created, we used Tukey’s HSD to give us clarity over where there are significant variances among mean prices for our independent variables.  The Tukey’s HSD outputs show that there is significant variation in the average prices of rentals.  As expected, whole homes/apartments cost more than private rooms and shared rooms, and private rooms cost more than shared rooms.  Likewise, as we expected Manhattan is significantly more costly than all other neighborhoods, with Brooklyn being the likely second most expensive neighborhood.  
	While this information does solve our objective of finding if there are significant variances across these dependent variables, we will take it a step further and design a model used to predict price based off all the dependent variables within the dataset.  To attempt to predict prices, we will use a stepwise linear model, k-nearest neighbors, bagging ensemble, and Random Forests.  We will use a benchmark predictor of the mean price for the whole data to compare our models to see if predictive modeling improves accuracy in decision-making.  After engineering and optimizing the various models, the models can be compared by the performance metrics (Figure 1.3).  Of all the models, the Random Forest was by far the best predictor of price.   It had more predictive power in accurately predicting new data and explained more % of variance in the data than any other model created.  
	To properly analyze which variables are the most important (i.e. which factors influence the pricing of AirBNB rentals), we can utilize the Random Forests Importance plot.  This plot shows the breakdown of what factor’s omission from the model leads to the greatest increase in MSE and which factors most greatly increase Node Purity.  Room type is the most important factor in explaining price, as its % Increase in MSE when excluded is over 35%, whereas no other factor was even above 30%.  Surprisingly, neighborhood group is on the lower end of prediction importance.  This can likely be attributed to the fact that, while Manhattan is the clear leader in average price, our ANOVA showed that there is not always a statistically significant variation in price between neighborhoods outside of Manhattan.  For example, Staten Island does not differ significantly from Brooklyn, Queens, or Bronx.  Queens and Bronx are also not significantly different from each other.  That said, there is reason to believe that while neighborhood group does have a clear winner, the remaining neighborhoods are not different enough in price to provide good value as predictors.  Room type, conversely, appears to be the absolute best factor across many models in predicting the rental price of an AirBNB rental.

NOTE: THIS PROJECT WAS PERFORMED AS A FINAL PROJECT FOR A GRADUATE LEVEL DATA MINING AND PREDICTIVE ANALYTICS CLASS.  PERMISSION FROM THE PROFESSOR OF THIS CLASS WAS OBTAINED PRIOR TO PUBLISHING.  
