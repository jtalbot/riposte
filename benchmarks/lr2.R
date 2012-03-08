# logistic regression test case

N <- 50000L
D <- 30L

p <- read.table("benchmarks/data/lr_p.txt")[[1]]
dim(p) <- c(N,D)
cat(length(p),'\n')

r <- read.table("benchmarks/data/lr_r.txt")[[1]]
cat(length(r),'\n')

wi <- read.table("benchmarks/data/lr_wi.txt")[[1]]

g <- function(z) 1/(1+exp(-z))

update <- function(w) {
	grad <- rep(0,D)
	diff <- g(p %*% w)-r
	for(i in 1L:D) {
		grad[i] <- mean((p[,i]*diff))
	}
	grad
}

benchmark <- function(reps) {

	w <- wi
	epsilon <- 0.07

	for(j in 1L:reps) {
		grad <- update(w)
		delta <- grad*epsilon
		w <- w - delta
	}
	
	w
	#glm(r~p-1, family=binomial(link="logit"), na.action=na.pass)
}

system.time(benchmark(3000L))
