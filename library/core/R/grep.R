
.grep <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    if(length(pattern)==0L)
        .stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        .stop("argument 'pattern' has length > 1")

    if(.isTRUE(perl)) {
        .stop("NYI: perl regex")
    }
    else {
        .Map('grep_map', 
            list(
                .External('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text)
            ), 'logical')
    }
}

grepl <- function(pattern, x, ignore.case, value, perl, fixed, useBytes, invert)
{
    # value argument is ignored

    r <- .grep(pattern, x, ignore.case, perl, fixed, useBytes)[[1]]
    
    if(.isTRUE(invert))
        r <- !r

    r
}

grep <- function(pattern, x, ignore.case, value, perl, fixed, useBytes, invert)
{
    r <- grepl(pattern, x, ignore.case, FALSE, perl, fixed, useBytes, invert)
    
    if(.isTRUE(value))
        x[r]
    else 
        seq_len(length(x))[r]
}

.regex <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    if(length(pattern)==0L)
        .stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        .stop("argument 'pattern' has length > 1")

    if(.isTRUE(perl)) {
        .stop("NYI: perl regex")
    }
    else {
        .Map('regex_map', 
            list(
                .External('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text)
            ), .characters('integer', 'integer'))
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
        .stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        .stop("argument 'pattern' has length > 1")

    if(.isTRUE(perl)) {
        .stop("NYI: perl regex")
    }
    else {
        .Map('gregex_map', 
            list(
                .External('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text)
            ), .characters('list', 'list'))
    }
}

gregexpr <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    m <- .gregex(pattern, text, ignore.case, perl, fixed, useBytes)
    r <- m[[1]]

    for(i in seq_len(length(r))) {
        attr(r[[i]], 'match.length') <- m[[2]][[i]]
        attr(r[[i]], 'useBytes') <- useBytes
    }
    
    r
}

sub <- function(pattern, replacement, text, ignore.case, perl, fixed, useBytes)
{
    .Map('sub_map',
        list(
            .External('regex_compile',
                as.character.default(pattern), 
                identical(ignore.case, TRUE),
                identical(fixed, TRUE)),
            as.character.default(text),
            as.character.default(replacement)
        ), .characters('character'))[[1]]
}

gsub <- function(pattern, replacement, text, ignore.case, perl, fixed, useBytes)
{
    .Map('gsub_map',
        list(
            .External('regex_compile',
                as.character.default(pattern), 
                identical(ignore.case, TRUE),
                identical(fixed, TRUE)),
            as.character.default(text),
            as.character.default(replacement)
        ), .characters('character'))[[1]]
}

