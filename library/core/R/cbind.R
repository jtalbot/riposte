
cbind <- function(deparse.level, ...) {
	l <- list(...)

    if(length(l) == 0L)
        return(NULL)
    
    f <- function(x) {
        if(is.matrix(x))
            as.integer(dim(x)[[1L]])
        else
            NA_integer_
    }

    rows <- .Map(f, list(l), 'integer')[[1L]]
    if(all(is.na(rows))) {
        nrow <- max(.Map(length, list(l), 'integer')[[1L]])
    }
    else {
        nrow <- min(rows[!is.na(rows)])
        if(nrow != max(rows[!is.na(rows)]))
            .stop("number of columns of matrices must match")
    }

    r <- function(x) {
        if(is.matrix(x))
            x
        else
            rep_len(x, nrow)
    }

	x <- unlist(.Map(r, list(l)),FALSE,FALSE)
    ncol <- length(x)/nrow
    
    dim(x) <- c(nrow, ncol)
    if(!is.null(names(l)) || !is.null(names(l[[1L]])))
        dimnames(x) <- list(rep_len(names(l[[1L]]), nrow), names(l))
	x
}

rbind <- function(deparse.level, ...) {
	l <- list(...)
    
    f <- function(x) {
        if(is.matrix(x))
            as.integer(dim(x)[[2L]])
        else
            NA_integer_
    }

    cols <- .Map(f, list(l), 'integer')[[1L]]
    if(all(is.na(cols))) {
        ncol <- max(.Map(length, list(l), 'integer')[[1L]])
    }
    else {
        ncol <- min(cols[!is.na(cols)])
        if(ncol != max(cols[!is.na(cols)]))
            .stop("number of columns of matrices must match")
    }

    r <- function(x) {
        if(is.matrix(x)) {
            # t
            nrow <- dim(x)[[1L]]
            x[nrow*(index(ncol,1L,length(x))-1L)+
                 index(nrow,ncol,length(x))]
        } else
            rep_len(x, ncol)
    }

	x <- unlist(.Map(r, list(l)),FALSE,FALSE)
    nrow <- length(x)%/%ncol
    # t
    x <- x[ncol*(index(nrow,1L,length(x))-1L)+
                 index(ncol,nrow,length(x))]

	dim(x) <- c(nrow, ncol)

    if(length(l) > 0L)
    {
        rownames <- names(l)
        if(is.matrix(l[[1]]))
            colnames <- dimnames(l[[1L]])[[2L]]
        else
            colnames <- names(l[[1L]])

        if(!is.null(rownames) || !is.null(colnames))
            dimnames(x) <- list(rownames, rep_len(colnames, ncol))
	}

    x
}

