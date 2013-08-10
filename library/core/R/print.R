
print.default <- function(x, digits, quote, na.print, print.gap, right, max, useSource, noOpt) 
{
    .cat(.format(x), '\n')
    .invisible(x)
}

print.function <- function(x, useSource, ...)
{
    .cat(.format.function(x), '\n')
    .invisible(x)
}

