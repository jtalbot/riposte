options(echo = FALSE)

# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org
# contributed by Saptarshi Guha and Justin Talbot
# modified from Mike Pall's Lua implementation

A <-  function(i, j){
  ij <- i+j-1
  1 / (ij*(ij-1)*0.5+i)
}

Av <- function(v,N){
  ma <- matrix(A(rep.int(1:N,N), rep.int(1:N, rep(N,N))), N, N)
  ma %*% v
}

Atv <- function(v,N){
  ma <- matrix(A(rep.int(1:N, rep(N,N)), rep.int(1:N, N)), N, N)
  ma %*% v
}

AtAv <- function(v, N){
  Atv(Av(v,N), N)
}

run <- function() {
	N <- as.numeric(commandArgs(TRUE)[1])

	u <- rep.int(1,N)
	for(i in 1:10){
          v <- AtAv(u,N)
	  u <- AtAv(v,N)
	}
	vBv <- sum(u*v)
	vv <- sum(v*v)
	cat(sqrt(vBv/vv), '\n')
}

system.time(run())
