
as.numeric <- as.double 

is.numeric <- function(x) .type(x) == 'double' || .type(x) == 'integer'

