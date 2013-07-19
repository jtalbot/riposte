
.grep <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    if(length(pattern)==0L)
        stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        stop("argument 'pattern' has length > 1")

    # TODO: respect useBytes
    
    if(identical(perl, TRUE)) {
        stop("NYI: perl regex")
    }
    else {
        .Map('grep_map', 
            list(
                .External(regex_compile(
                    as.character(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE))),
                as.character(text)
            ), 'logical')
    }
}

grepl <- function(pattern, x, ignore.case, value, perl, fixed, useBytes, invert)
{
    # value argument is ignored

    r <- .grep(pattern, x, ignore.case, perl, fixed, useBytes)[[1]]
    
    if(identical(invert, TRUE))
        r <- !r

    r
}

grep <- function(pattern, x, ignore.case, value, perl, fixed, useBytes, invert)
{
    r <- grepl(pattern, x, ignore.case, FALSE, perl, fixed, useBytes, invert)
    
    if(identical(value, TRUE))
        x[r]
    else 
        seq_len(length(x))[r]
}

.regex <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    if(length(pattern)==0L)
        stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        stop("argument 'pattern' has length > 1")

    # TODO: respect useBytes
    
    if(identical(perl, TRUE)) {
        stop("NYI: perl regex")
    }
    else {
        .Map('regex_map', 
            list(
                .External(regex_compile(
                    as.character(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE))),
                as.character(text)
            ), c('integer', 'integer'))
    }
}

regexpr <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    m <- .regex(pattern, text, ignore.case, perl, fixed, useBytes)
    r <- m[[1]]
    attr(r, 'match.length') <- m[[2]]
    attr(r, 'useBytes') <- useBytes
    r
}

.gregex <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    if(length(pattern)==0L)
        stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        stop("argument 'pattern' has length > 1")

    # TODO: respect useBytes
    
    if(identical(perl, TRUE)) {
        stop("NYI: perl regex")
    }
    else {
        .Map('gregex_map', 
            list(
                .External(regex_compile(
                    as.character(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE))),
                as.character(text)
            ), c('list', 'list'))
    }
}

gregexpr <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    m <- .gregex(pattern, text, ignore.case, perl, fixed, useBytes)
    r <- m[[1]]

    # TODO: replace with an lapply
    for(i in seq_len(length(r))) {
        attr(r[[i]], 'match.length') <- m[[2]][[i]]
        attr(r[[i]], 'useBytes') <- useBytes
    }
    
    r
}

sub <- function(pattern, replacement, text, ignore.case, perl, fixed, useBytes)
{
    # TODO: sub needs to be able to use submatches
    m <- regexpr(pattern, text, ignore.case, perl, fixed, useBytes)
    
    start <- strip(m)
    start <- pmin(pmax(as.integer(start)-1L, 0L), nchar(text))
    
    length <- attr(m, 'match.length')

    .Map('substrassign_map', 
        list(text, start, length, replacement), 'character')[[1]]
}
