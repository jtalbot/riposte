## R version of convolution example from extensions manual
convolve <- function(a, b) # from the extending R manual
{
    ab <- double(length(a) + length(a))

    for(i in 1:length(a)) {
        for(j in 1:length(b)) {
            ab[j+i] <- ab[j+i] + a[i]*b[j]
        }
    }
    ab
}

x <- as.double(1:500)

#convolve(x,x)
system.time(convolve(x,x))
