#CHL7001 Applied Machine Learning for Health Data - Assignment 1
#Name: Ivy Fong (1000735145)
#Date: October 28, 2018

#set working directory
setwd("C:/Users/ivyfo/Dropbox/Master of Public Health/Master of Public Health - Courses/Fall 2018 - Courses/CHL7001 - Machine Learning/CHL7001 - Assignments/CHL7001 A1")

#load package to be used
library(glmnet)


##Task 1 - Read the data into R, making sure that you code the missing values properly. The character "?" is used for denoting missing values in the .csv file. Notice that there is no header in the data file. (1 point)
bc <- read.csv("bc_data.csv", header=F, na.strings="?") #read csv data into R, specify no variable names, return ? for missing values
bc <- na.omit(bc) #only keep observations with complete information


##Task 2 - Report descriptive statistics and make a box plot for the time variable for the two groups (with and without recurrence). Make a subset of the original dataset with only those without recurrence. (2 points) 
bcn <- subset(bc, bc$V2=="N") #create subset with only those without recurrence
bcr <- subset(bc, bc$V2=="R") #create subset with only those with recurrence

mean(bcn$V3) #calculate mean of time variable for those without recurrence = 53.58108
sd(bcn$V3) #calculate sd of time variable for those without recurrence = 34.91935

mean(bcr$V3) #calculate mean of time variable for those with recurrence = 25.56522
sd(bcr$V3) #calculate sd of time variable for those with recurrence = 22.72703

#Time - Group Without Recurrence (N):
  #Mean = 53.58108
  #SD = 34.91935

#Time - Group With Recurrence (R):
  #Mean = 25.56522
  #SD = 22.72703

boxplot(bc$V3~bc$V2, xlab="Recurrence Status", ylab="Time") #create boxplot for the time variable for the two groups 


##Task 3 - Using as predictors the mean values of the above described (a) - (j) features (which as found in columns 4-13), train a ridge regression model using the default grid of values for the lambda parameter in the glmnet R function. Make a plot showing the coefficients of these predictors for different levels of regularization. Comment on the results. (3 points)
x <- model.matrix(V3~V4+V5+V6+V7+V8+V9+V10+V11+V12+V13,bcn) #specify the predictor variables V4-13
y <- bcn$V3 #specify the outcome variable V3

rr.mod <- glmnet(x,y,family="gaussian",alpha=0) #train a ridge regression model using the default values
plot(rr.mod, xvar="lambda") #plot the coefficients for different levels of regularization

#Comment: As the value of log lambda increases, the coefficients either increase or decrease to approach 0.


##Task 4 - Using a 5-fold cross-validation estimate and report the optimal value for lambda (i.e. that minimizes the MSE). Make a plot showing the MSE against the values of log(lambda). Report the coefficients of the predictors for the optimal lambda value. (3 points)
cv.rr <- cv.glmnet(x,y,alpha=0,nfolds=5) #perform 5-fold cross-validation 
plot(cv.rr) #plot MSE versus log(lambda)
cv.rr$lambda.min #print the lambda associated with the min MSE = 13.37345
coef.min <- coef(cv.rr, s = "lambda.min") #print the unstandardized coefficients associated with the optimal lambda 
coef.min

#optimal lambda value = 13.37345
#predictor coefficients for the optimal lambda value: 
  #V4  =   -0.401431533
  #V5  =   -1.489660254
  #V6  =   -0.086699448
  #V7  =   -0.003732896
  #V8  =  177.468091758
  #V9  =  -13.158970300
  #V10 =  -48.074388890
  #V11 =  -11.901168233
  #V12 =   69.336268882
  #V13 =  485.974704557


##Task 5 - Calculate the MSE on the whole set of the non-recurrent group for the model using the optimal lambda value. (2 points)
mean((y-predict(rr.mod, newx=x, s= cv.rr$lambda.min))^2) #calculate the MSE of the whole data set using the optimal lamda = 985.8034

#MSE using the optimal lambda value = 985.8034


##Task 6 - Repeat tasks 3-5 above but this time using the lasso method. This time report also what the selected features are for the optimal lambda value. (7 points)
l.mod <- glmnet(x,y,family="gaussian",alpha=1) #train a lasso model using the default values
plot(l.mod, xvar="lambda") #plot the coefficients for different levels of regularization

#Comment: As the value of log lambda increases, the coefficients either increase or decrease to 0.

cv.l <- cv.glmnet(x,y,alpha=1,nfolds=5) #perform 5-fold cross-validation 
plot(cv.l) #plot MSE versus log(lambda)
cv.l$lambda.min #print the lambda associated with the min MSE = 0.7373547
coef.min2 <- coef(cv.l, s = "lambda.min") #print the unstandardized coefficients associated with the optimal lambda
coef.min2
row.names(coef.min2)[as.vector(coef.min2)!=0] #print only non-zero coefficients = V5, V6, V8, V10, V12, V13

#optimal lambda value = 0.7373547
#predictor coefficients for the optimal lambda value: 
  #V4  =    .         
  #V5  =   -1.85388150
  #V6  =   -0.06579758
  #V7  =    .         
  #V8  =  224.85581587
  #V9  =    .         
  #V10 = -104.62778262
  #V11 =    .         
  #V12 =   52.30459849
  #V13 =  826.01279857
#selected features for the optimal lambda value = V5, V6, V8, V10, V12, V13

mean((y-predict(l.mod, newx=x, s= cv.l$lambda.min))^2) #calculate the MSE of the whole data set using the optimal lamda = 972.7765

#MSE using the optimal lambda value = 972.7765


##Task 7 - Make some comments on how the two methods compare based on the analysis you did and the results you generated above. Suggest a rigorous method or approach of actually comparing the performance of the two prediction methods on these data. You do not have to apply this comparison method. (2 points)
#Based on the results generated above, the lasso method seems to have performed better because it has a lower MSE using the optimal lambda value.
#Lasso tends to perform better than ridge regression when only some features are strong predictors, as is the case here - only 6/10 features were selected.
#MSE (ridge regression) = 985.8034, MSE (lasso) = 972.7765.

#The model comparison process must be separate from the parameter tuning process, and so different training, validation and test sets must be used.
#Thus, to compare the performance of the two prediction methods on these data, we could use CV, LOO, validation set, or nested cross-validation methods.
