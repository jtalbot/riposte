
sign <- function(x) UseGroupMethod('sign', 'Math', x)
sign.default <- function(x) (x>0)-(x<0)

