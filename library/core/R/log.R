
log <- function(x, base=exp(1)) UseGroupMethod('log', 'Math', x)
log.default <- function(x, base=exp(1)) 
    .ArithUnary2('log_map', x)/.ArithUnary2('log_map', base)

log10 <- function(x) UseGroupMethod('log10', 'Math', x)
log10.default <- function(x) log(x, 10)

log2 <- function(x) UseGroupMethod('log2', 'Math', x)
log2.default <- function(x) log(x, 2)

# TODO: compute in a numerically stable manner
log1p <- function(x) UseGroupMethod('log1p', 'Math', x)
log1p.default <- function(x) log(1+x)


exp <- function(x) UseGroupMethod('exp', 'Math', x)
exp.default <- function(x) .ArithUnary2('exp_map', x)

# TODO: compute in a numerically stable manner
expm1 <- function(x) UseGroupMethod('expm1', 'Math', x)
expm1.default <- function(x) exp(x)-1

