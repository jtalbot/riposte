# logistic regression test case

{

library("compiler")
enableJIT(3)

M <- as.integer(commandArgs(TRUE)[[2]])
N <- as.integer(commandArgs(TRUE)[[1]])/M
D <- 30L

p <- read.table("data/lr_p.txt")[[1]]
dim(p) <- c(N,D)

r <- read.table("data/lr_r.txt")[[1]]
wi <- read.table("data/lr_wi.txt")[[1]]

g <- function(z) 1/(1+exp(-z))

update <- function(w) {
	diff <- g(p %*% w)-r
	grad <- double(0)
	for(i in 1L:D) {
		grad[[i]] <- mean((p[,i]*diff))
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

cat(system.time(benchmark(M))[[3]])

}
