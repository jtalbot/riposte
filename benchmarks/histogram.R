
#data <- sample(1:100, 100000, replace=TRUE)

data <- floor(runif(10000000,0,100))

#length(split(data,data,100L))
system.time(length(split(data,data,100L)))

