# logistic regression test case

x <- read.table("tests/lr_x2.csv")
x <- c(rep(1,1,4000),x)
cat(length(x), "\n")
y <- read.table("tests/lr_y2.csv")
cat(length(y), "\n")


g <- function(z) 1/(1+exp(-z))
w <- c(0,0,0,0)

i <- rep(4, 4000, 16000)
k <- rep(4000,1,16000)+1

z <- x
dim(z) <- c(4000,4)

update <- function(w) sum(split((g(z %*% w)-y)[k]*x, i, 4))/4000
epsilon <- 0.07
for(j in 1:100000) {
	grad <- update(w)
	delta <- grad*epsilon
	w <- w - delta
}

w
