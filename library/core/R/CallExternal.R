
.C <- function(.NAME, ..., NAOK=FALSE, DUP=TRUE, PACKAGE=NULL, ENCODING=NULL) {
    if(is.character(.NAME)) {
        .NAME <- getSymbolInfo(.NAME, PACKAGE, FALSE)
    
        if(is.null(.NAME))
            .stop(sprintf("Can't find .C function: %s", .NAME))
    }

    print(sprintf(".C: %s", .NAME$name))
    if(!inherits(.NAME, 'NativeSymbolInfo', FALSE)) {
        .stop("Unknown .NAME argument to .Call")
    }
    
    .Riposte('dotC', .NAME$address, list(...))
}

.Call <- function(.NAME, ..., PACKAGE=NULL) {
    if(is.character(.NAME)) {
        .NAME <- getSymbolInfo(.NAME, PACKAGE, FALSE)
        
        if(is.null(.NAME))
            .stop(sprintf("Can't find .Call function: %s", .NAME))
    }
    
    print(sprintf(".Call: %s", .NAME$name))
    if(!inherits(.NAME, 'NativeSymbolInfo', FALSE)) {
        .stop("Unknown .NAME argument to .Call")
    }

    .Riposte('dotCall', .NAME$address, list(...))
}

.External <- function(.NAME, ..., PACKAGE=NULL) {
    if(is.character(.NAME)) {
        .NAME <- getSymbolInfo(.NAME, PACKAGE, FALSE)
        
        if(is.null(.NAME))
            .stop(sprintf("Can't find .Riposte function: %s", .NAME))
    }
    
    print(sprintf(".External: %s", .NAME$name))
    if(!inherits(.NAME, 'NativeSymbolInfo', FALSE)) {
        .stop("Unknown .NAME argument to .Riposte")
    }

    .Riposte('dotExternal', .NAME$address, list(.NAME, ...))
}

