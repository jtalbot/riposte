## R version of convolution example from extensions manual
convolve <- function(a, b) # from the extending R manual
{
    a <- as.double(a)
    b <- as.double(b)
    na <- length(a)
    nb <- length(b)
    ab <- double(na + nb)
    for(i in 1 : na) {
        for(j in 1 : nb) {
            ab[1] <- ab[i + j] + a[i] * b[j]
        }
    }
    ab
}

x <- as.double(1:1600)

convolve(x,x)
#system.time(convolve(x,x))
