
beta <- function(a, b) .ArithBinary2('beta_map', a, b)
lbeta <- function(a, b) .ArithBinary2('lbeta_map', a, b)

gamma <- function(x) UseGroupMethod('gamma', 'Math', x)
gamma.default <- function(x) .ArithUnary2('gamma_map', x)

lgamma <- function(x) UseGroupMethod('lgamma', 'Math', x)
lgamma.default <- function(x) .ArithUnary2('lgamma_map', x)

digamma <- function(x) UseGroupMethod('digamma', 'Math', x)
digamma.default <- function(x) .ArithUnary2('digamma_map', x)

trigamma <- function(x) UseGroupMethod('trigamma', 'Math', x)
trigamma.default <- function(x) .ArithUnary2('trigamma_map', x)

psigamma <- function(x, deriv) .ArithBinary2('psigamma_map', x, deriv)

choose <- function(n, k) .ArithBinary2('choose_map', n, k)
lchoose <- function(n, k) .ArithBinary2('lchoose_map', n, k)

