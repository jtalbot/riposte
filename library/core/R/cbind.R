
cbind <- function(deparse.level, ...) {
	l <- list(...)

    if(length(l) == 0L)
        return(NULL)
    
    f <- function(x) {
        if(is.matrix(x))
            dim(x)[[1L]]
        else
            NA_integer_
    }

    rows <- as.integer(lapply(l, f))
    if(all(is.na(rows))) {
        nrow <- max(as.integer(lapply(l, length)))
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

	x <- unlist(lapply(l, r),FALSE,FALSE)
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
            dim(x)[[2L]]
        else
            NA_integer_
    }

    cols <- as.integer(lapply(l, f))
    if(all(is.na(cols))) {
        ncol <- max(as.integer(lapply(l, length)))
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

	x <- unlist(lapply(l, r),FALSE,FALSE)
    nrow <- length(x)/ncol
    # t
    x <- x[ncol*(index(nrow,1L,length(x))-1L)+
                 index(ncol,nrow,length(x))]

	dim(x) <- c(nrow, ncol)
    if(!is.null(names(l)) || !is.null(names(l[[1L]])))
        dimnames(x) <- list(names(l), rep_len(names(l[[1L]]), ncol))
	x
}

