
.C <- function(.NAME, ..., NAOK=FALSE, DUP=TRUE, PACKAGE=NULL, ENCODING=NULL) {
    if(is.character(.NAME)) {
        .NAME <- getSymbolInfo(.NAME, PACKAGE, FALSE)
    
        if(is.null(.NAME))
            .stop(sprintf("Can't find .C function: %s", .NAME))
    }

    if(!inherits(.NAME, 'NativeSymbolInfo', FALSE)) {
        .stop("Unknown .NAME argument to .Call")
    }
    
    .External('dotC', .NAME, list(...))
}

.Call <- function(.NAME, ..., PACKAGE=NULL) {
    if(is.character(.NAME)) {
        .NAME <- getSymbolInfo(.NAME, PACKAGE, FALSE)
        
        if(is.null(.NAME))
            .stop(sprintf("Can't find .Call function: %s", .NAME))
    }
    
    if(!inherits(.NAME, 'NativeSymbolInfo', FALSE)) {
        .stop("Unknown .NAME argument to .Call")
    }

    print('.Call')
    print(.NAME)
}

