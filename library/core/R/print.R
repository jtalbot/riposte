
.format.vector <- function(x, type)
{
    if(length(x) == 0)
        cat(.External(paste(list(type, '(0)'), "")))
    else
        cat(x)
}

print <- function(x, ...) UseMethod("print")

print.default <- function(x)
{
    cat(typeof(x))
}

print.logical <- function(x)
{
    cat(.External(print(x)))
}

print.integer <- function(x)
{
    cat(.External(print(x)))
}

print.double <- function(x)
{
    cat(.External(print(x)))
}

print.character <- function(x)
{
    #x <- ifelse(is.na(x), "NA", paste(list('"', .escape(x), '"'), "", NULL))
    #.format.vector(x, 'character')
    cat(.External(print(x)))
}

print.list <- function(x)
{
    cat(.External(print(x)))
}

print.environment <- function(x)
{
    cat(.External(print(x)))
}

print.NULL <- function(x)
{
    cat("NULL")
}

print.function <- function(x)
{
    cat(.External(print(x)))
}

