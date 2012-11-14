fib <- function(n) {
    if(n<=2)
        1
    else
        fib(n-1)+fib(n-2)
}

system.time(fib(20))
