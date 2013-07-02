
print.default <- function(x, digits, quote, na.print, print.gap, right, max, useSource, noOpt) 
{
    text <- format(x)
    cat(text, '\n')
    x
}

print.function <- function(x, useSource, ...)
{
    text <- format.function(x)
    cat(text, '\n')
    x
}

