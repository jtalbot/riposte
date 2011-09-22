x <- 1:100000000
a <- 0
add <- function() {
system.time(for(i in x) a <<- a+x[i])
}
add()
