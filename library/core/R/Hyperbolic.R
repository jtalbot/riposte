
cosh <- function(x) UseGroupMethod('cosh', 'Math', x)
cosh.default <- function(x) .ArithUnary2('cosh_map', x)

sinh <- function(x) UseGroupMethod('sinh', 'Math', x)
sinh.default <- function(x) .ArithUnary2('sinh_map', x)

tanh <- function(x) UseGroupMethod('tanh', 'Math', x)
tanh.default <- function(x) .ArithUnary2('tanh_map', x)


acosh <- function(x) UseGroupMethod('acosh', 'Math', x)
acosh.default <- function(x) .ArithUnary2('acosh_map', x)

asinh <- function(x) UseGroupMethod('asinh', 'Math', x)
asinh.default <- function(x) .ArithUnary2('asinh_map', x)

atanh <- function(x) UseGroupMethod('atanh', 'Math', x)
atanh.default <- function(x) .ArithUnary2('atanh_map', x)

