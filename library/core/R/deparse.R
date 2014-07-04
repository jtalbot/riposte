
deparse <- function(expr, width.cutoff, backtick, control, nlines) {
    .deparse(expr)
}

.deparse <- function(x,...) {
    if(is.nil(x))
        ''
    else
        UseMethod(".deparse", x)
}

.wrap <- function(x, type, attrs) {
    if(length(x) == 0L)
        r <- sprintf("%s(0)", type)
    else if(length(x) == 1L)
        r <- x
    else
        r <- .concat(list('c(', paste(x,sep=', ', collapse=', '), ')'))

    if(!is.null(attrs)) {
        r <- .pconcat('structure(', r)
        n <- names(attrs)
        for(i in seq_len(length(attrs))) {
            r <- .pconcat(r, ', ')
            k <- ''
            if(!is.null(n) && n[[i]] != '')
                k <- .pconcat(n[[i]], ' = ')
            k <- .pconcat(k, .deparse(attrs[[i]]))
        
            r <- .pconcat(r, k)
        }
        r <- .pconcat(r, ')')
    }
    r
}

.deparse.default <- function(x)
{
    .wrap(.deparse(strip(x)), '', attributes(x))
}

.deparse.logical <- function(x)
{
    attrs <- attributes(x)
    x <- strip(x)
    .wrap(
        ifelse(is.na(x),
            'NA',
            ifelse(x, 'TRUE', 'FALSE')),
        'logical',
        attrs)
}

.deparse.integer <- function(x)
{
    attrs <- attributes(x)
    x <- strip(x)
    if(length(x) > 1L && !any(is.na(x)) && 
        .isTRUE(all(x == x[[1]]:(x[[1]]+length(x)-1L))))
        .wrap(sprintf('%dL:%dL', x[[1]], x[[1]]+length(x)-1L),
                'integer',
                attrs)
    else
        .wrap(
            ifelse(is.na(x),
                'NA_integer_', 
                sprintf("%lldL", x)),
                'integer',
                attrs)
}

.deparse.double <- function(x)
{
    attrs <- attributes(x)
    x <- strip(x)
    d <- .Map('decimals_map', list(x, 15L), 'integer')[[1]]
    format <- ifelse(d >= 0, '%.*f', '%.*e')
    .wrap(
        ifelse(is.infinite(x),
            ifelse(x>0, 'Inf', '-Inf'),
            ifelse(is.nan(x),
                'NaN',
                ifelse(is.na(x),
                    'NA_real_',
                    ifelse(x,
                        sprintf(format, ifelse(d>=0,d,-d-1L), x),
                        '0'                # handles the -0 case
        )))),
        'double',
        attrs)
}

.deparse.character <- function(x)
{
    attrs <- attributes(x)
    x <- strip(x)
    .wrap(
        ifelse(is.na(x),
            'NA_character_', 
            sprintf("\"%s\"", x)),
        'character',
        attrs)
}

.deparse.pairlist <- .deparse.list <- function(x)
{
    n <- names(x)
    attrs <- attributes(x)
    x <- strip(x)
    r <- 'list('
    for(i in seq_len(length(x))) {
        k <- ''
        if(!is.null(n) && n[[i]] != '')
            k <- .pconcat(n[[i]], ' = ')
        k <- .pconcat(k, .deparse(x[[i]]))
        
        r <- .pconcat(r, k)

        if(i < length(x))
            r <- .pconcat(r, ', ')
    }
    r <- .pconcat(r, ')')
    .wrap(r, 'list', attrs)
}

.deparse.environment <- function(x)
{
    # R could do a lot better here...
    '<environment>'
}

.deparse.NULL <- function(x)
{
    .wrap('NULL', 'NULL', attributes(x))
}

.deparse.function <- function(f)
{
    x <- f[['formals']]
    n <- names(x)
    x <- strip(x)
    r <- 'function('
    for(i in seq_len(length(x))) {
        k <- ''
        if(!is.null(n) && n[[i]] != '')
            k <- .pconcat(n[[i]], ' = ')
        if(!is.na(x[[i]]))
            k <- .pconcat(k, .deparse(x[[i]]))
        
        r <- .pconcat(r, k)

        if(i < length(x))
            r <- .pconcat(r, ', ')
    }
    r <- .pconcat(r, ') ')
    r <- .pconcat(r, .deparse(f[['body']]))
    .wrap(r , '', attributes(x))
}

.deparse.name <- function(x)
{
    strip(x)
}

.deparse.indexing <- function(x, open, close) {
    if(length(x) >= 2)
        func <- .pconcat(.deparse(x[[2]]), open)
    else
        func <- .pconcat('NULL', open)
    
    if(length(x) >= 3) {
        for(i in 2L+seq_len(length(x)-2L)) {
        
            func <- .pconcat(func, .deparse(x[[i]]))
            if(i < length(x))
                func <- .pconcat(func, ', ')
        }
    }
    .pconcat(func, close)
}

.deparse.call <- function(x, ...)
{
    if(length(x) == 0) {
        "NULL"
    }
    else {
        e <- x[[1]]
        func <- .deparse(e)
        n <- names(x)

        if(length(x) == 2 && (func == '+' || func == '-' || func == '!' || func == '~')
            && is.null(n)) {
            .pconcat(func, .deparse(x[[2]]))
        }
        else if(length(x) == 3 && (
            func == '/' ||
            func == '%/%' ||
            func == '^' ||
            func == '%%' ||
            func == '$' ||
            func == '@' ||
            func == ':' ||
            func == '::' ||
            func == ':::')
            && is.null(n)) {
            .concat(list(.deparse(x[[2]]), func, .deparse(x[[3]])))
        }
        else if(length(x) == 3 && (
            func == '+' ||
            func == '-' ||
            func == '*' ||
            func == '~' ||
            func == '&' ||
            func == '|' ||
            func == '&&' ||
            func == '||' ||
            func == '==' ||
            func == '!=' ||
            func == '<' ||
            func == '<=' ||
            func == '>' ||
            func == '>=' ||
            func == '<-' ||
            func == '<<-')
            && is.null(n)) {
            .concat(list(.deparse(x[[2]]), func, .deparse(x[[3]])))
        }
        else if(func == '[') {
            .deparse.indexing(x, '[', ']') 
        }
        else if(func == '[[') {
            .deparse.indexing(x, '[[', ']]') 
        }
        else if(func == '(' && is.null(n) && length(x) == 2) {
            func <- '('
            func <- .pconcat(func, .deparse(x[[2]]))
            .pconcat(func, ')')
        }
        else if(func == '{' && is.null(n)) {
            func <- '{\n'
            for(i in 1L+seq_len(length(x)-1L)) {
                func <- .pconcat(func, .pconcat('\t',.deparse(x[[i]])))
                func <- .pconcat(func,'\n')
            }
            .pconcat(func, '}')
        }
        else if(func == 'function' && is.null(n)) {
            if(is.character(x[[4]]))
                x[[4]][[1]]
            else {
                n <- names(x[[2]])
                a <- strip(x[[2]])
                args <- ''
                for(i in seq_len(length(a))) {
                    k <- n[[i]]
                    if(!is.nil(a[[i]]))
                        k <- .pconcat(.pconcat(k, '='), a[[i]])
                    args <- .pconcat(args, k)
                    if(i < length(a))
                        args <- .pconcat(args, ', ')
                }
                .pconcat(.pconcat(.pconcat(.pconcat(func, '('), args), ') '), .deparse(x[[3]])) 
            }
        }
        else if(func == 'if' && is.null(n)) {
            r <- .pconcat(.pconcat(.pconcat('if (', .deparse(x[[2]])), ') '), .deparse(x[[3]]))
            if(length(x) == 4) {
                r <- .pconcat(r, .pconcat('else ', .deparse(x[[4]])))
            }
            r
        }
        else if(is.null(n)) {
            func <- .pconcat(func, '(')
            for(i in 1L+seq_len(length(x)-1L)) {
                func <- .pconcat(func, .deparse(x[[i]]))
                if(i < length(x))
                    func <- .pconcat(func, ', ')
            }
            .pconcat(func, ')')
        }
        else {
            func <- .pconcat(func, '(')
            for(i in 1L+seq_len(length(x)-1L)) {
                a <- .deparse(x[[i]])
                if(n[[i]] != '')
                    a <- .pconcat(.pconcat(n[[i]], ' = '), a)
                func <- .pconcat(func, a)
                if(i < length(x))
                    func <- .pconcat(func, ', ')
            }
            .pconcat(func, ')')
        }
    }
}

.deparse.expression <- function(x, ...)
{
    func <- 'expression(' 
    if(length(x) > 0) {
        for(i in seq_len(length(x))) {
            func <- .pconcat(func, .deparse(x[[i]]))
            if(i < length(x))
               func <- .pconcat(func, ', ')
        }
    }
    .pconcat(func, ')')
}


