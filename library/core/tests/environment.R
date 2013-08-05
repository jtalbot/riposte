
{
a <- new.env(FALSE, emptyenv(), 1)
b <- new.env(FALSE, a, 1)

a[['a']] <- 10

f <- function() a

environment(f) <- b

f()
}

{

(f <- function (x) 
parent.frame(x))
f(1)

# DIFF: R treats the global environment as its own parent, Riposte doesn't yet and may never do so.
#f(2)

(g <- function (x) 
{
    f <- function(x) parent.frame(x)
    f(x)
})
g(2)

# DIFF: R treats the global environment as its own parent, Riposte doesn't yet and may never do so.
#g(3)

}
