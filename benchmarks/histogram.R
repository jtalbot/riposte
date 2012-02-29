
data <- sample(1:100, 100000, replace=TRUE)

lapply(split(data,data,100), length)

