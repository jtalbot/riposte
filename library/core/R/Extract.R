
`[` <- function(x, ...) UseMethod('[', x)

`[.default` <- function(x, ..., drop = TRUE)
{
    if(...() == 0L)
        return(x)
    if(...() != 1L)
        .stop("incorrect number of dimensions")
    
    i <- ..1
    if(is.character(i)) {
        i <- .semijoin(i, as.character(names(x)))
        i[i==0L] <- length(x)+1
    }
    else if(is.null(i)) {
        i <- vector('integer',0)
    }

    if(is.integer(i) || is.double(i) || is.logical(i)) {
        r <- strip(x)[strip(i)]
        # discard all attributes but e.g. names
        if(!is.null(attr(x,'names'))) {
            attr(r,'names') <- attr(x,'names')[strip(i)]
        }
        r
    }
    else {
        .stop("invalid subscript type")
    }
}

`[.environment` <- function(x, i, ...) {
    if(!is.character(i))
        .stop('wrong arguments for subsetting an environment')
    strip(x)[[strip(i)]]
}

#`[` <- function(x, ..., drop = TRUE) {
#	i <- list(...)
#	d <- dim(x)
#	i[is.na(i)] <- lapply(d[is.na(i)], function(x) 1:x)
#	d <- cumprod(c(1,d[-length(d)]))
#	i <- mapply(function(i) (i[[1]]-1)*i[[2]], i, d)
#	nd <- as.integer(lapply(i, function(x) length(x)))
#	len <- prod(nd)
#	a <- cumprod(nd)
#}

`[.matrix` <- `[.array` <- function(x, ..., drop = TRUE)
{
    if(...() == 0L)
        return(x)

    d <- attr(x, 'dim')
    
    if((length(d) != 1L || identical(drop,TRUE)) && ...() == 1L)
        return( `[.default`(x,..1) )

    if(length(d) != ...())
        .stop("incorrect number of dimensions")

    idx <- list()
    for(i in seq_len(...())) {
        if(missing(...(i))) {
            idx[[i]] <- seq_len(d[[i]])
        }
        else { 
            id <- strip(...(i))

            if(is.na(id))
                idx[[i]] <- rep_len(NA, d[[i]])
            else if(is.null(id))
                idx[[i]] <- vector('integer',0)
            else if(is.logical(id))
                idx[[i]] <- which(id)
            else if(is.character(id)) {
                if(is.null(attr(x, 'dimnames')))
                    .stop("no 'dimnames' attribute for array")
                r <- .semijoin(id, as.character(attr(x,'dimnames'))[[i]])
                if(any(r==0L))
                    .stop("subscript out of bounds")
                idx[[i]] <- r
            }
            else if(is.integer(id) || is.double(id))
                idx[[i]] <- as.integer(id)
            else
                .stop(sprintf("invalid subscript type '%s'", class(id)))
        }
    }

    mult <- 1
    indices <- idx[[1]]*mult
    for(i in seq_len(length(d)-1L)) {
        mult <- mult * d[[i]]
        indices <- rep.int(indices, length(idx[[i+1L]]))
        indices <- indices + (idx[[i+1L]]-1L)*mult
    }

    r <- strip(x)[indices]
    attr(r,'dim') <- as.integer(lapply(idx, function(x) length(x)))

    dn <- dimnames.default(x)
    if(!is.null(dn)) {
        rn <- list()
        for(i in seq_len(length(dn))) {
            rn[[i]] <- dn[[i]][idx[[i]]]
        }
        dimnames(r) <- rn
    }
    r
}

`[[` <- function(x, ..., exact = TRUE) {
    UseMethod('[[', x)
}

#`[[` <- function(x, ..., exact = TRUE) {
#	i <- as.integer(list(...))
#	
#	d <- dim(x)
#	if(is.null(d)) d <- length(x)
#	
#	if(length(i) != length(d)) .stop("incorrect number of subscripts")
#	if(any(i < 1) || any(i > d)) .stop("subscript out of bounds")
#
#	if(length(d) > 1) {
#		d <- c(1,d[-length(d)])
#		d <- sum((i-1)*cumprod(d))+1
#	}
#	strip(x)[[d]]
#}

`[[.default` <- function(x, i, ..., exact = TRUE) {
    if(is.character(i)) {
        i <- which(names(x)==i)
        if(length(i)==0)
            .stop("subscript out of bounds") 
       
        strip(x)[[ i[[1]] ]]
    }
    else if(is.integer(i) || is.double(i)) {
	    strip(x)[[strip(i)]]
    }
    else {
        .stop("invalid subscript type")
    }
}

`[[.call` <- `[[.expression` <- `[[.list` <- `[[.pairlist` <- 
    function(x, i, ..., exact = TRUE) {
    if(is.character(i)) {
        i <- which(names(x)==i)
        if(length(i)==0)
            NULL
        else 
            strip(x)[[ i[[1]] ]]
    }
    else if(is.integer(i) || is.double(i)) {
	    strip(x)[[strip(i)]]
    }
    else {
        .stop("invalid subscript type")
    }
}

`[[.environment` <- function(x, i, ...) {
    if(!is.character(i) || length(i) != 1L)
        .stop('wrong arguments for subsetting an environment')
    strip(x)[[strip(i)]]
}

`[[.closure` <- function(x, i, ...) {
    strip(x)[[strip(i)]]
}


`[<-` <- function(x, ..., value) UseMethod('[<-', x)

.copy.most.attributes <- function(r, x, i, nn) {
    i <- strip(i)
    # copy over attributes, taking care to keep names lined up
    for(n in names(attributes(x))) {
        a <- attr(x,n)
        if(n == 'names') {
            a[i] <- ifelse(is.na(a[i]), nn, a[i])
            a[is.na(a)] <- ''
        }
        attr(r,n) <- a
    }
    r
}

`[<-.default` <- function(x, i, ..., value) {
    nn <- ''
    if(is.character(i)) {
        i <- nn <- strip(i)
        i <- .semijoin(i, as.character(names(x)))
        i[i==0L] <- length(x)+seq_len(sum(i==0L))
        
        if(is.null(attr(x,'names')))
            attr(x,'names') <- rep('', length(x))
    }
    else if(is.null(i)) {
        i <- vector('integer',0)
    }

    if(is.integer(i) || is.double(i) || is.logical(i)) {
        r <- `[<-`(strip(x), strip(i), value)
        .copy.most.attributes(r, x, i, nn)
    }
    else {
        .stop('invalid subscript type')
    }
}

`[<-.environment` <- function(x, i, ..., value) {
    if(is.character(i))
        `[<-`(strip(x), strip(i), value)
    else
        .stop('wrong args for environment subassignment')
    x
}

`[[<-` <- function(x, ..., value) {
    UseMethod('[[<-', x)
}

`[[<-.default` <- function(x, i, ..., value) {
    nn <- ''
    if(is.character(i)) {
        i <- nn <- strip(i)
        i <- which(names(x) == i)
        if(length(i)==0)
            i <- length(x)+1
        else
            i <- i[[1]]

        if(is.null(attr(x,'names'))) 
            attr(x,'names') <- rep('', length(x))
    }
    
    if(is.integer(i) || is.double(i)) {
        r <- `[[<-`(strip(x), strip(i), value)
        .copy.most.attributes(r, x, i, nn)
    }
    else {
        .stop('invalid subscript type')
    }
}

`[[<-.environment` <- function(x, i, ..., value) {
    if(is.character(i))
        `[[<-`(strip(x), strip(i), value)
    else
        .stop('wrong args for environment subassignment')
    x
}

`[[<-.closure` <- function(x, i, ..., value) {
    .stop('Assignment to function members is not yet implemented')
}

`$` <- function(a, b) {
    a[[strip(.pr_expr(.getenv(NULL), quote(b)))]]
}

`$<-` <- function(x, i, value) {
    x[[strip(.pr_expr(.getenv(NULL), quote(i)))]] <- value
    x
}

