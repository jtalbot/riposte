
readDCF <- function(file, fields, keep.white) {
    lines <- readLines(file, -1, TRUE, FALSE, 'unknown')
    split <- strip(regexpr(':', lines, FALSE, FALSE, TRUE, FALSE))
    keys <- substr(lines, 0L, split-1L)
    values <- substr(lines, split+1L, .nchar(lines))

    if(is.null(fields)) {
        fields <- unique(keys, NULL, FALSE, length(keys))
        fields <- fields[fields != '']
    }

    r <- vector('character', 0)
    dim(r) <- c(0L, length(fields))
    row <- rep_len(NA_character_, length(fields))
    some <- FALSE
    for(i in seq_len(length(lines))) {
        if(keys[[i]] == '') {
            if(some) {
                r <- rbind(0L, r, row)
            }
            some <- FALSE
            row <- rep_len(NA_character_, length(fields))
        }
        else {
            idx <- .semijoin(keys[[i]], fields)
            if(idx > 0L) {
                row[idx] <- values[[i]]
                some <- TRUE
            }
        }
    }
    if(some) {
        r <- rbind(0L, r, row)
    }
    dimnames(r) <- list(NULL, fields)
    r
}

