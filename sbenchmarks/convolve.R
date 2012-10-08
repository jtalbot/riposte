## R version of convolution example from extensions manual
convolve <- function(a, b) # from the extending R manual
{
    ab <- double(length(a) + length(a))

    js <- 1L:length(a)
    for(i in 1L:length(a)) {
        for(j in js) {
            ab[j+i] <- ab[j+i] + a[i]*b[j]
        }
    }
    ab
}

x <- as.double(1:20000)

#convolve(x,x)
system.time(convolve(x,x))
