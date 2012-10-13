## R version of convolution example from extensions manual
convolve <- function(a, b) # from the extending R manual
{
    ab <- double(length(a) + length(a))

    for(i in 1L:length(a)) {
        for(j in 1L:length(b)) {
            ab[j+i] <- ab[j+i] + a[i]*b[j]
        }
    }
    ab
}

N <- as.integer(sqrt(as.integer(commandArgs(TRUE)[[1]])))
x <- as.double(1:N)

#cat(convolve(x,x))
cat(system.time(convolve(x,x))[[3]])
