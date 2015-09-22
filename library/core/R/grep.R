
.grep <- function(pattern, text, ignore.case, perl, fixed, useBytes)
{
    if(length(pattern)==0L)
        .stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        .stop("argument 'pattern' has length > 1")

    if(fixed === TRUE || perl !== TRUE) {
        .Map('grep_map',
            list(
                .Riposte('regex_compile',
                    as.character.default(pattern),
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text)
            ), 'logical')
    }
    else {
        .Map('pcre_grep_map',
            list(
                .Riposte('pcre_regex_compile',
                    as.character.default(pattern),
                    identical(ignore.case, TRUE)),
                as.character.default(text)
            ), 'logical')
    }
}

grepl <- function(pattern, x, ignore.case, value, perl, fixed, useBytes, invert)
{
    # value argument is ignored

    r <- .grep(pattern, x, ignore.case, perl, fixed, useBytes)[[1]]
    
    if(invert === TRUE)
        r <- !r

    r
}

grep <- function(pattern, x, ignore.case, value, perl, fixed, useBytes, invert)
{
    r <- grepl(pattern, x, ignore.case, FALSE, perl, fixed, useBytes, invert)
    
    if(value === TRUE)
        x[r]
    else 
        seq_len(length(x))[r]
}

.agrep <- function(pattern, text, ignore.case, costs, bounds, useBytes, fixed)
{
    if(length(pattern)==0L)
        .stop("invalid 'pattern' argument")
    if(length(pattern)>1L)
        .stop("argument 'pattern' has length > 1")

    .Map('agrep_map',
        list(
            .Riposte('regex_compile',
                as.character.default(pattern),
                identical(ignore.case, TRUE),
                identical(fixed, TRUE)),
            as.character.default(text),
            list(as.integer(costs)),
            list(as.double(bounds)),
            list(length(pattern))
        ), 'logical')
}

agrepl <- function(pattern, x, ignore.case, value, costs, bounds, useBytes, fixed)
{
    # value argument is ignored
    .agrep(pattern, x, ignore.case, costs, bounds, useBytes, fixed)[[1]]
}

agrep <- function(pattern, x, ignore.case, value, costs, bounds, useBytes, fixed)
{
    r <- agrepl(pattern, x, ignore.case, FALSE, costs, bounds, useBytes, fixed)

    if(value === TRUE)
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

    if(fixed === TRUE || perl !== TRUE) {
        .Map('regex_map', 
            list(
                .Riposte('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text)
            ), .characters('integer', 'integer'))
    }
    else {
        .Map('pcre_regex_map', 
            list(
                .Riposte('pcre_regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE)),
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

    if(fixed === TRUE || perl !== TRUE) {
        .Map('gregex_map', 
            list(
                .Riposte('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text)
            ), .characters('list', 'list'))
    }
    else {
        .Map('pcre_gregex_map', 
            list(
                .Riposte('pcre_regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE)),
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
    if(fixed === TRUE || perl !== TRUE) {
        .Map('sub_map',
            list(
                .Riposte('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text),
                as.character.default(replacement)
            ), .characters('character'))[[1]]
    }
    else {
        .Map('pcre_sub_map',
            list(
                .Riposte('pcre_regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text),
                as.character.default(replacement)
            ), .characters('character'))[[1]]
    }
}

gsub <- function(pattern, replacement, text, ignore.case, perl, fixed, useBytes)
{
    if(fixed === TRUE || perl !== TRUE) {
        .Map('gsub_map',
            list(
                .Riposte('regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text),
                as.character.default(replacement)
            ), .characters('character'))[[1]]
    }
    else {
        .Map('pcre_gsub_map',
            list(
                .Riposte('pcre_regex_compile',
                    as.character.default(pattern), 
                    identical(ignore.case, TRUE),
                    identical(fixed, TRUE)),
                as.character.default(text),
                as.character.default(replacement)
            ), .characters('character'))[[1]]
    }
}

