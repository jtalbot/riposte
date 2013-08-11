
normalizePath <- function(path, winslash, mustWork) {
    r <- .Map('realpath_map', 
        list(as.character.default(path), as.character.default(winslash)), c('character'))[[1]]
    
    mustWork <- as.logical(mustWork)
    if(.isTRUE(mustWork) && any(is.na(r)))
        .stop("No such file or directory")
    else if(is.na(mustWork) && any(is.na(r)))
        warning(TRUE, FALSE, "No such file or directory")

    r
}

