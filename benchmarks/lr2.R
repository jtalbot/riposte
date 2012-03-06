# logistic regression test case

N <- 50000L
D <- 30L

p <- read.table("benchmarks/data/lr_p.txt")
dim(p) <- c(N,D)
print(length(p),'\n')

r <- read.table("benchmarks/data/lr_r.txt")
print(length(r),'\n')

wi <- read.table("benchmarks/data/lr_wi.txt")

g <- function(z) 1/(1+exp(-z))

mv <- function(m, v) {
	r <- 0
	for(i in 1L:ncol(m)) {
		r <- r + m[,i]*v[[i]]	
	}
	r
}

update <- function(w) {
	grad <- rep(0,D)
	diff <- g(mv(p,w))-r
	for(i in 1L:D) {
		grad[i] <- mean((p[,i]*diff))
	}
	grad
}

#i <- rep(4, 4000, 16000)
#k <- rep(4000,1,16000)+1
#update <- function(w) sum(split((g(z %*% w)-y)[k]*x, i, 4))/4000

benchmark <- function(reps) {

	w <- wi
	epsilon <- 0.07

	for(j in 1L:reps) {
		grad <- update(w)
		delta <- grad*epsilon
		w <- w - delta
	}
	
	w
}

system.time(benchmark(3000L))
