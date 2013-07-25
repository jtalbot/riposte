
abs <- function(x) UseGroupMethod('abs', 'Math', x)
abs.default <- function(x) .ArithUnary1('fabs_map', 'iabs_map', x)

sqrt <- function(x) UseGroupMethod('sqrt', 'Math', x)
sqrt.default <- function(x) .ArithUnary2('sqrt_map', x)

