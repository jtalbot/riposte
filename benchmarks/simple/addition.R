add <- function() {
x <- 1:100000000
a <- 0
system.time(for(i in x) a <- a+x[i])
}
add()
