setwd("~/Desktop/hw02")
data_set <- read.csv("hw02_data_set_images.csv",header = FALSE)
Y_truth_label <- read.csv("hw02_data_set_labels.csv",header = FALSE)
Y_truth_label <- Y_truth_label$V1
safelog <- function(x) {
  return (log(x + 1e-100))
}

sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

train_a <- data_set[1:25,]
train_b <- data_set[40:64,]
train_c <- data_set[79:103,]
train_d <- data_set[118:142,]
train_e <- data_set[157:181,]
test_a <- data_set[26:39,]
test_b <- data_set[65:78,]
test_c <- data_set[104:117,]
test_d <- data_set[143:156,]
test_e <- data_set[182:195,]

training_set <- rbind(train_a,train_b,train_c,train_d,train_e)
test_set <- rbind(test_a,test_b,test_c,test_d,test_e)

K <- 5L
N <- length(Y_truth_label)

Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, Y_truth_label)] <- 1

Y_truth_test <- cbind(Y_truth[26:39,],Y_truth[65:78,],Y_truth[104:117,],Y_truth[143:156,],Y_truth[182:195,])
dim(Y_truth_test) <- c(70,5)
Y_truth_train <- cbind(Y_truth[1:25,],Y_truth[40:64,],Y_truth[79:103,],Y_truth[118:142,],Y_truth[157:181,])
dim(Y_truth_train) <- c(125,5)

gradient_W <- function(X, Y_truth, Y_predicted) {
  return (sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix((Y_truth[,c] - Y_predicted[,c]) * Y_predicted[,c] * (1-Y_predicted[,c]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
}

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (colSums((Y_truth - Y_predicted) * Y_predicted * (1-Y_predicted)))
}

eta <- 0.01
epsilon <- 1e-3

set.seed(521)
W <- matrix(runif(ncol(training_set) * K, min = -0.01, max = 0.01), ncol(training_set), K)
w0 <- runif(K, min = -0.01, max = 0.01)


# data setten hesaplama 
training_set <- data.matrix(training_set)
test_set <- data.matrix(test_set)
Y_truth_train <- data.matrix(Y_truth_train)
iteration <- 1
objective_values <- c()
while (1) {
 # Y_predicted <- sigmoid(t(W) * training_set + w0)
  Y_predicted <- t(sapply(X = 1:125, FUN = function(c) {sigmoid(t(W) %*% training_set[c,] + w0)}))
  #objective_values <- c(objective_values, -sum(Y_truth * safelog(Y_predicted + 1e-100)))
  objective_values <- c(objective_values,sum((1/2)*(Y_truth_train - Y_predicted)^2))
  
  W_old <- W
  w0_old <- w0
  
  W <- W + eta * gradient_W(training_set, Y_truth_train, Y_predicted)
  w0 <- w0 + eta * gradient_w0(Y_truth_train, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((W - W_old)^2)) < epsilon ) {
    break
  }
  
  iteration <- iteration + 1
}

print(W)
print(w0)
print(iteration)
plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

conf_truth_train <- rowSums(sapply(X=1:5, FUN=function(c) {Y_truth_train[,c]*c}))
y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix_train <- table(y_predicted, conf_truth_train)
print(confusion_matrix_train)

Y_predicted_test <- t(sapply(X = 1:70, FUN = function(c) {sigmoid(t(W) %*% test_set[c,] + w0)}))

conf_truth_test <- rowSums(sapply(X=1:5, FUN=function(c) {Y_truth_test[,c]*c}))
Y_predicted_test <- apply(Y_predicted_test, 1, which.max)
confusion_matrix_test <- table(Y_predicted_test, conf_truth_test)
print(confusion_matrix_test)
