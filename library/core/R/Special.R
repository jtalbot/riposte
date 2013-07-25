
beta <- function(a, b)
    .stop("beta (NYI)")
lbeta <- function(a, b)
    .stop("lbeta (NYI)")

gamma <- function(x) UseGroupMethod('gamma', 'Math', x)
gamma.default <- function(x)
    .stop("gamma (NYI)")

lgamma <- function(x) UseGroupMethod('lgamma', 'Math', x)
lgamma.default <- function(x)
    .stop("lgamma (NYI)")

digamma <- function(x) UseGroupMethod('digamma', 'Math', x)
digamma.default <- function(x)
    .stop("digamma (NYI)")

trigamma <- function(x) UseGroupMethod('trigamma', 'Math', x)
trigamma.default <- function(x)
    .stop("trigamma (NYI)")

psigamma <- function(x, deriv)
    .stop("psigamma (NYI)")

choose <- function(n, k)
    .stop("choose (NYI)")
lchoose <- function(n, k)
    .stop("lchoose (NYI)")

