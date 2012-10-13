
Posdef <- function (n, ev = runif(n, 0, 10))
{
Z <- matrix(ncol=n, rnorm(n^2))
decomp <- qr(Z)
Q <- qr.Q(decomp)
R <- qr.R(decomp)
d <- diag(R)
ph <- d / abs(d)
O <- Q %*% diag(ph)
Z <- t(O) %*% diag(ev) %*% O
return(Z)
}

N <- as.integer(commandArgs(TRUE)[[1]])
a <- Posdef(as.integer(sqrt(N)))

write.table(as.vector(a), "data/cholesky.txt", col.names=FALSE, row.names=FALSE)
