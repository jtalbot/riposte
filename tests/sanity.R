fail <- 0

{
	cat("Testing if statement\n-------------\n")
	if(TRUE) cat("PASS\n") else cat ("FAIL\n")
	if(FALSE) cat("FAIL\n") else cat ("PASS\n")
	if(1) cat("PASS\n") else cat("FAIL\n")
	if(is.null(if(FALSE) 1)) cat("PASS\n") else cat("FAIL\n")
	"done"
}

{
	PassIfTrue <- function(x) { fail<-fail+1; if(x) { cat("PASS\n"); fail<-fail-1 } else cat("FAIL\n") }
	PassIfFalse <- function(x) { fail<-fail+1; if(x) cat("FAIL\n") else { cat("PASS\n"); fail<-fail-1 } }
	"defined PassIfTrue and PassIfFalse"
}

{
	cat("Testing basic equality of built in types\n--------------------------------\n")
	PassIfTrue(1 == 1)
	PassIfFalse(1 == 0)
	PassIfTrue(1L == 1L)
	PassIfFalse(1L == 0L)
	PassIfTrue('a' == 'a')
	PassIfFalse('a' == 'b')
	PassIfTrue(TRUE == TRUE)
	PassIfFalse(TRUE == FALSE)
	PassIfTrue(1+2i == 1+2i)
	PassIfFalse(1+2i == 1+3i)
	"done"
}

{
	cat("Testing math coercion\n-------------------------\n")
	PassIfTrue(1 == 1L)
	PassIfFalse(1 == 0L)
	PassIfTrue(1 == TRUE)
	PassIfFalse(1 == FALSE)
	"done"
}

{
	cat("Testing basic math ops\n-----------------------\n")
	PassIfTrue(0 + 0 == 0)
	PassIfTrue(0 + 1 == 1)
	PassIfTrue(+1 == 1)
	PassIfTrue(0 - 5 == -5)
	PassIfTrue(-5 == -5)
	PassIfTrue(0 * 1 == 0)
	PassIfTrue(2 * 1 == 2)
	PassIfTrue(4 / 2 == 2)
	PassIfTrue(1 / 1 == 1)
	PassIfTrue(2 %/% 2 == 1)
	PassIfTrue(3 %/% 2 == 1)
	PassIfFalse(5 %/% 2 == 1)
	PassIfTrue(5 %% 2 == 1)
	PassIfTrue(6 %% 2 == 0)
	PassIfTrue(2 ^ 2 == 4)
	PassIfTrue(3 ^ 4 == 81)
	"done"
}

{
	cat("Testing loop constructs\n-----------------------\n")
	a <- 0
	for(i in 1:10) a <- a+1
	PassIfTrue(a == 10)
	PassIfTrue(i == 10)
	a <- 1:10
	for(i in a) 1
	PassIfTrue(i == 10)

	a <- 0
	while(a < 10) a <- a+1
	PassIfTrue(a == 10)
	
	a <- 0
	repeat {
		if(a == 10) break
		else a <- a+1
	}
	PassIfTrue(a == 10)
	
	"done"
}

{
	cat("Testing parameter passing rules\n------------------------\n")
	f <- function(x)
	a <- 1
	PassIfTrue(f(1) == 1)
	PassIfTrue(f(a) == 1)
	PassIfTrue(f(a) == f(1))

	cat("Name matching...\n")
	f <- function(x, y) x/y
	PassIfTrue(f(2,1) == 2)
	PassIfTrue(f(x=2,1) == 2)
	PassIfTrue(f(2,x=1) == 0.5)
	PassIfTrue(f(2,y=1) == 2)
	PassIfTrue(f(y=2,1) == 0.5)

	cat("Partial name matching...\n")
	f <- function(first, second) first^second
	PassIfTrue(f(2,3) == 8)
	PassIfTrue(f(first=2,3) == 8)
	PassIfTrue(f(f=2,3) == 8)
	PassIfTrue(f(3,f=2) == 8)
	PassIfTrue(f(sec=3,2) == 8)
	
	cat("Dots...\n")
	f <- function(x,...) x
	g <- function(...) f(...)
	h <- function(...) f(100,...)
	i <- function(k,...) f(...,k)

	PassIfTrue(f(1,2,3) == 1)
	PassIfTrue(g(1,2,3) == 1)
	PassIfTrue(h(1,2,3) == 100)
	PassIfTrue(i(1,2,3) == 2)
	
	"done"
}

{
	cat("Testing some basic vector operations\n-----------------------\n")
	a <- c(1,2,3)
	PassIfTrue(a[1] == 1)
	PassIfTrue(a[3] == 3)
	PassIfTrue((a+1)[1] == 2)
	PassIfTrue((a+1)[3] == 4)
	PassIfTrue(sum(a) == 6)
	
	b <- c(TRUE, FALSE)
	PassIfTrue(b[1])
	PassIfFalse(b[2])
	PassIfFalse(b[[2]])

	l <- list(a, b, 4)
	PassIfTrue(l[[3]] == 4)
	PassIfTrue(l[[1]][1] == 1)

	"done"
}

#can it print out a function?
f <- function(x,y) x+y

{
	cat("Tricky bugs\n-----------------------\n")
	f <- function(x, y) {z <- 0; x+z}
	PassIfTrue(f(4,5) == 4) 
	z <- 10
	f <- function(x, y) {z <- 0; rm(z); x+z}
	PassIfTrue(f(4,5) == 14)
}


if(fail == 0)
	"SUCCESS! All sanity checks passed"
else
	"FAILURE! Some sanity checks failed"
