
as.integer <- function(x, ...) as(strip(x), 'integer')

is.integer <- function(x) .type(x) == 'integer'

