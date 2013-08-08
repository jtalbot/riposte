
fib <- function(x) if(x <= 1L) 1L else fib(x-1L)+fib(x-2L)

a <- fib

fib <- function(x) 1L

a(5)

Recall <- function(...) .Internal(Recall(...))

fib <- function(x) if(x <= 1L) 1L else Recall(x-1L)+Recall(x-2L)

a <- fib

fib <- function(x) 1

a(5)

