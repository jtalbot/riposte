
.ArithUnary1 <- function(ffunc, ifunc, x) {
    switch(.type(x),
        double=.Map.double(ffunc, x),
        integer=,
        logical=.Map.integer(ifunc, as.integer(x)),
        stop("Non-numeric argument to mathematical function"))
}

abs <- function(x) .ArithUnary1('fabs_map', 'iabs_map', x)


.ArithUnary2 <- function(func, x) {
    switch(.type(x),
        double=.Map.double(func, x),
        integer=,
        logical=.Map.double(func, as.double(x)),
        stop("Non-numeric argument to mathematical function"))
}

sqrt <- function(x) .ArithUnary2('sqrt_map', x)
floor <- function(x) .ArithUnary2('floor_map', x)
ceiling <- function(x) .ArithUnary2('ceiling_map', x)
trunc <- function(x) .ArithUnary2('trunc_map', x)

exp <- function(x) .ArithUnary2('exp_map', x)
log <- function(x) .ArithUnary2('log_map', x)

cos <- function(x) .ArithUnary2('cos_map', x)
sin <- function(x) .ArithUnary2('sin_map', x)
tan <- function(x) .ArithUnary2('tan_map', x)

acos <- function(x) .ArithUnary2('acos_map', x)
asin <- function(x) .ArithUnary2('asin_map', x)
atan <- function(x) .ArithUnary2('atan_map', x)

atan2_i <- function(x, y) .Map.double('atan2_map', as.double(x), as.double(y))


nchar <- function(x, type = "chars", allowNA = FALSE) {
    .Map.integer('nchar_map', as.character(x))
}

nzchar <- function(x) .Map.logical('nzchar_map', as.character(x))

.escape <- function(x) {
    .Map.character("escape_map", as.character(x))
}

.pconcat <- function(x,y) {
    .Map.character("concat_map", as.character(x), as.character(y))
}

.concat <- function(x) {
    .Fold.character("concat", as.character(x))
}

.OrdinalUnary <- function(func, x) {
    switch(.type(x),
        double=.Map.logical(func, x),
        integer=,
        logical=,
        NULL=.Map.logical(func, as.double(x)),
        stop("Non-numeric argument to mathematical function"))
}

is.nan <- function(x) {
    x != x
}

is.finite <- function(x) {
    if(is.double(x))
        (abs(x) != Inf) & (x == x)
    else
        x == x
}

is.infinite <- function(x) {
    if(is.double(x))
        (abs(x) == Inf) & (x == x)
    else
        x != x
}

.ArithBinary2 <- function(func, x, y) {
    x <- switch(.type(x),
        double=x,
        integer=,
        logical=,
        NULL=as.double(x),
        stop("Non-numeric argument to mathematical function"))

    y <- switch(.type(y),
        double=y,
        integer=,
        logical=,
        NULL=as.double(y),
        stop("Non-numeric argument to mathematical function"))

    .Map.double(func, x, y)
}

atan2 <- function(x, y) .ArithBinary2('atan2_map', x, y)

hypot <- function(x, y) .ArithBinary2('hypot_map', x, y)

.Digits <- function(func, x, y) {
    x <- switch(.type(x),
        double=x,
        integer=,
        logical=,
        NULL=as.double(x),
        stop("Non-numeric argument to mathematical function"))

    y <- switch(.type(y),
        double=,
        integer=,
        logical=,
        NULL=as.integer(y),
        stop("Non-numeric argument to mathematical function"))

    .Map.double(func, x, y)
}

round <- function(x, digits = 0) .Digits('round_map', x, digits)

signif <- function(x, digits = 6) .Digits('signif', x, digits)


.ArithUnary2 <- function(func, x) {
    switch(.type(x),
        double=.Map.double(func, x),
        integer=,
        logical=.Map.double(func, as.double(x)),
        stop("Non-numeric argument to mathematical function"))
}

mean <- function(x) sum(x)/length(x)

cummean <- function(x) .Scan.double('mean', x)
