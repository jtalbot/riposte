# from R mailing list, Joerg van den Hoff, 25 Aug 2006
fib <- function(n) { 
	if (n < 2) 
		fn <- 1
	else 
		fn <- fib(n - 1) + fib(n - 2)
	fn
}
fib(30)
