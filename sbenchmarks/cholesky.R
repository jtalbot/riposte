
N <- as.integer(commandArgs(TRUE)[[1]])
N <- as.integer(sqrt(N))

m <- read.table("data/cholesky.txt")[[1]]
dim(m) <- c(N,N)

run <- function(m, N) {
    for(i in 1:(N-1)) {
        col <- m[i:N,i]
        col <- col/sqrt(col[1])
        m[i:N,i] <- col
        m[(i+1):N,(i+1):N] <- m[(i+1):N,(i+1):N] - col[-1] %o% col[-1]
    }
    m[N,N] <- sqrt(m[N,N])
    m
}

cat(system.time(run(m, N))[[3]])
