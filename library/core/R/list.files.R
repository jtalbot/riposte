
list.files <- function(path, pattern, all.files, full.names, recursive, ignore.case, include.dirs, no..) {
    r <- .Map('listfiles_map', list(path.expand(path)), 
        'list')[[1]]

    if(!is.null(pattern)) {
        for(i in seq_along(r)) {
            r[[i]] <- grep(pattern, r[[i]], ignore.case, TRUE, FALSE, FALSE, FALSE, FALSE)
        }
    }

    if(full.names) {
        for(i in seq_along(r)) {
            r[[i]] <- .pconcat(.pconcat(path[[i]],'/'), r[[i]])
        }
    }

    r <- unlist(r, FALSE, FALSE)

    r
}

