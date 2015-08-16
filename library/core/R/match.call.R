
.match.call <- function(x, table) {
    
    # find complete matches
    r <- match(x, table, 0L, '')
    complete <- rep(FALSE, length(table))

    for(i in seq_len(length(r))) {
        if(r[[i]] != 0L) {
            if(complete[r[[i]]])
                .stop(.concat(list(
                    'formal argument "',
                    table[r[[i]]],
                    '" matched by multiple actual arguments')))
            else
                complete[r[[i]]] <- TRUE
        }
    }

    # find partial matches (up to the dots) 
    dots <- match('...', table, length(table)+1L, NULL)
    complete <- complete[seq_len(dots-1L)]

    xtable <- table[seq_len(dots-1L)]
    partial <- rep(FALSE, length(xtable))
     
    for(i in seq_len(length(x))) {
        if(r[[i]] == 0L && x[[i]] != '') {
            m <- (substr(xtable,1L,nchar(x[[i]])) == x[[i]])
            m <- m & !complete

            if(sum(m) > 1L)
                r[[i]] <- NA_integer_
            else if(sum(m) == 1L) {
                j <- which(m)
                if(partial[[j]] == FALSE) {
                    partial[[j]] <- TRUE
                    r[[i]] <- j
                }
                else {
                    r[r==j] <- NA_integer_
                }
            }
        }
    }

    # assign unassigned
    unassigned <- which(!(partial | complete))
    unmatched <- which(r==0L & x == '')
  
    k <- pmin(length(unassigned), length(unmatched))

    r[unmatched[seq_len(k)]] <- unassigned[seq_len(k)]

    r
}

match.call <- function(definition, call, expand.dots, envir)
{
    if(is.null(definition))
        definition <- .frame(2L)[['.__function__.']]

    # If the call has dots, expand the call using the dots in the
    # enclosing scope.
    # NOTE: doing this really only makes sense if match.call is called
    # with its defaults.
    d <- 0L
    for(i in seq_len(length(call)))
        if(identical(strip(call[[i]]), '...'))
            d <- i
    if(d > 0L) {
        promise('dots', quote(list(...)), .frame(3L), .getenv(NULL))
        call <- c(call[seq_len(d-1L)], dots, call[d+seq_len(length(call)-d)])
    }
    
    f <- formals(definition)
    
    args <- call[-1L]
    argnames <- names(args)
    if(is.null(argnames))
        argnames <- rep('', length(args))
    m <- .match.call(argnames, names(f))

    if(any(is.na(m)))
        .stop('argument is ambiguous')

    f[m] <- args

    indots <- any(m==0L)
    if(indots) {
        dots <- match('...', names(f), 0L, NULL)
        if(dots == 0L)
            .stop('unused argument')
    }
    else
        dots <- 0L

    # drop any unneeded arguments
    f <- f[which(match(seq_len(length(f)), c(m,dots), 0L, NULL)>0L)]
    dots <- match('...', names(f), 0L, NULL)
    if(dots != 0L) {
        if(identical(expand.dots, TRUE)) {
            f <- c( f[seq_len(dots-1L)], 
                    args[m==0L], 
                    f[dots+seq_len(length(f)-dots)] )
        }
        else {
            f[[dots]] <- args[m==0L]
        }
    }
   
    as.call(c(call[[1L]], f))
}

