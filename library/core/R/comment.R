
comment <- function(x) attr(x, 'comment')

`comment<-` <- function(x, value) {
    if(length(value) == 0L)
        attr(x, 'comment') <- NULL
    else if(!is.character(value))
        .stop("attempt to set invalid 'comment' attribute")
    else
        attr(x, 'comment') <- value
    x
}

