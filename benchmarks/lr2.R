# logistic regression test case

`-.default` <- function(x,y) {
	if(missing(y)) {
		-strip(x)
		#attributes(r) <- attributes(x)
	} else {
		strip(x) - strip(y)
		#attributes(r) <- attributes(x)
	}
}

x <- read.table("tests/lr_x2.csv")
x <- c(rep(1,1,4000),x)
cat(length(x), "\n")
y <- read.table("tests/lr_y2.csv")
cat(length(y), "\n")
g <- function(z) 1/(1+exp(-z))
w <- c(0,0,0,0)
i <- as.integer(0:15999 %/% 4000)
j <- as.integer(0:15999 %/% 4)
k <- as.integer(1:4000)
k <- c(k,k,k,k)

z <- x
dim(z) <- c(4000,4)

#update <- function(w) sum(split((g(sum(split(x*w[i], j, 400)))-y)[k]*x, i, 4))/400

#update <- function(w) sum(split((g(sum(split(x * w[i], j, 400)))-y)[k]*x, i, 4))/400
update <- function(w) sum(split((g(z %*% w)-y)[k]*x, i, 4))/4000
epsilon <- 0.07
for(j in 1:100000) {
	grad <- update(w)
	delta <- grad*epsilon
	w <- w - delta
}

w
