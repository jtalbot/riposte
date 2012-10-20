
N <- as.integer(commandArgs(TRUE)[[1]])
N <- as.integer(sqrt(N))

m <- read.table("data/cholesky.txt")[[1]]
dim(m) <- c(N,N)

run <- function(m, N) {
    for(i in 1:(N-1)) {
        cat(m, '\n')
        cat(dim(m), '\n')
        cat(i, '\n')
        col <- m[i:N,i]
        cat(i, '\n')
        col <- col/sqrt(col[[1]])
        cat(i, '\n')
        m[i:N,i] <- col
        #m[(i+1):N,(i+1):N] <- m[(i+1):N,(i+1):N] - col[2:length(col)] %o% col[2:length(col)]
    }
    m[N,N] <- sqrt(m[N,N])
    m
}

cat(system.time(run(m, N))[[3]])
