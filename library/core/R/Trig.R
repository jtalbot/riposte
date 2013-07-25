
cos <- function(x) UseGroupMethod('cos', 'Math', x)
cos.default <- function(x) .ArithUnary2('cos_map', x)

sin <- function(x) UseGroupMethod('sin', 'Math', x)
sin.default <- function(x) .ArithUnary2('sin_map', x)

tan <- function(x) UseGroupMethod('tan', 'Math', x)
tan.default <- function(x) .ArithUnary2('tan_map', x)


acos <- function(x) UseGroupMethod('acos', 'Math', x)
acos.default <- function(x) .ArithUnary2('acos_map', x)

asin <- function(x) UseGroupMethod('asin', 'Math', x)
asin.default <- function(x) .ArithUnary2('asin_map', x)

atan <- function(x) UseGroupMethod('atan', 'Math', x)
atan.default <- function(x) .ArithUnary2('atan_map', x)

atan2 <- function(y, x) .ArithBinary2('atan2_map', y, x)

