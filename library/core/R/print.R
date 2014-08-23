
print.default <- function(x, digits, quote, na.print, print.gap, right, max, useSource, noOpt) 
{
    cat(.pconcat(.format(x),'\n'), '', '', FALSE, NULL, NULL)
    .invisible(x)
}

print.function <- function(x, useSource, ...)
{
    cat(.pconcat(.format.function(x),'\n'), '', '', FALSE, NULL, NULL)
    .invisible(x)
}

