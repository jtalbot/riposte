x <- 1:100
a <- 0
add <- function() {
system.time(for(i in x) a <<- a+x[i])
}
add()
