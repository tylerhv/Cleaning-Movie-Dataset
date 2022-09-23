The goal of this was to clean the data as much as possible such that it was workable. 
My first goal was to remove all of the TV shows from the data set.
Then, change the VOTES and Gross column into floats and integers, since they were originally objects.

I wanted to see if there was a correlation between the number of votes a movie gets, and the amount of money a movie makes (Gross value)
I ran a linear regression, and created a distribution graph to see how well the model worked.
Then, we used a R^2 test to see how well the model worked. The number came out to be .41, or 41%. This means about 41% of the data points can be explained by this model. There is a weak correlation here.

Finally, we used a "out of testing" validation using R^2, to see how accurate our data was. The numbers came out to be -.014, .229, -9.884. This suggests that the linear regression model that we used may not be the best fit for this data set, or there is just a very low correlation between the number of votes a movie gets and how much it makes.
