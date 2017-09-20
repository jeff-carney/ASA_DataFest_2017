# omitting any NA's, maybe a better way to deal with NAs?
dataNoNa= na.omit(data)


tempData = dataNoNa[!duplicated(dataNoNa[,"user_id"]),]

# indices from main df for those who booked non-packages and those who booked packages
noPackBookInd = which(tempData$is_package == 0 & tempData$is_booking == 1)
packBookInd = which(tempData$is_package == 1 & tempData$is_booking == 1)

# getting a random sample of 60,000 indices from package and non-package indices
sampPackBookInd = sample(packBookInd, 9000)
sampNoPackBookInd = sample(noPackBookInd, 9000)

# combining indices from package and non-package and then sorting
testTrain = c(sampPackBookInd, sampNoPackBookInd)
testTrain = sort(testTrain)

# use sorted indices to extract data of interest
testTrain = tempData[testTrain,]

# convert to data frame
testTrain = data.frame(testTrain)

# get only relevant columns
testTrain = testTrain[,c("user_location_latitude", "user_location_longitude", "orig_destination_distance", "is_mobile", "is_package","channel", "srch_adults_cnt", "srch_children_cnt", "srch_rm_cnt", "srch_destination_id")]

# converting is_package to two-level factor variable for our training and testing labels
testTrain$is_package = factor(testTrain$is_package)

# getting all columns that are numeric
num.vars <- sapply(testTrain, is.numeric)

# normalizing all numeric columns
testTrain[num.vars] <- lapply(testTrain[num.vars], scale)

# size of test set
testSize = 0.3*nrow(testTrain)

set.seed(123) 

# test indices
test <- 1:testSize

# indexing out training and test sets
train.df <- testTrain[-test,]
test.df <- testTrain[test,]

# grabbing training and testing labels
train.lab <- testTrain$is_package[-test]
test.lab <- testTrain$is_package[test]

library(class)


# training model
knn.10 <-  knn(train.df, test.df, train.lab, k=10)
knn.50 <-  knn(train.df, test.df, train.lab, k=50)
knn.100 <- knn(train.df, test.df, train.lab, k=100)
knn.200 <- knn(train.df, test.df, train.lab, k=200)

# proportion predicted correctly
sum(test.lab == knn.10)/length(test.lab)
sum(test.lab == knn.50)/length(test.lab)
sum(test.lab == knn.100)/length(test.lab)
sum(test.lab == knn.200)/length(test.lab)

# initializing vector of k's and accuracy
ks = (1:300)
acc = rep(0, 300)

# trains model using k = 1:300 and adds accuracy to accuracy vector
for (i in 1:length(ks)){
    knn <- knn(train.df, test.df, train.lab, k = ks[i])
    acc[i] = sum(test.lab == knn)/length(test.lab)
    print(i)
}

# plots k vs. accuracy
plot(ks, acc, xlab="k (for kNN)", ylab="Accuracy")

# libraries for caret
library(e1071)
library(caret)

# training using cross validation to pick optimal k
double_check <- train(train.df, train.lab, method = "knn", 
    tuneGrid = data.frame(k = c(5, 7, 10, 15, 20))) 

# predictions using optimal k
pred = predict(double_check, newdata = test.df)

# accuracy on test set using optimal k
sum(pred == test.lab)/length(test.lab)
