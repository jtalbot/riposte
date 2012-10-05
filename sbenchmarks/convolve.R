## R version of convolution example from extensions manual
convolve <- function(a, b) # from the extending R manual
{
    a <- as.double(a)
    b <- as.double(b)
    na <- length(a)
    nb <- length(b)
    ab <- double(na + nb)

    for(i in 1:na) {
        for(j in 1:nb) {
            1 #ab[j+i] <- ab[j+i] + a[i]*b[j]
        }
    }
    ab
}

x <- as.double(1:10)

#convolve(x,x)
system.time(convolve(x,x))
