x <- 1:10000000
add <- function() {
for(i in x) 1
}
system.time(add())
