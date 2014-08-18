
loadedDLLs <- vector('list',0) 
attr(loadedDLLs, 'class') <- 'DLLInfoList'

getLoadedDLLs <- function() {
    loadedDLLs
}

dyn.load <- function(x, local, now, str) {
    name <- x
    handle <- NULL
    info <- list(`.C`=list(), `.Call`=list(), `.Fortran`=list(), `.Riposte`=list(), useDynamicLookup=FALSE, forceSymbols=FALSE)
        
    if(x != 'base') {
        handle <- .Riposte('dynload', as.character(x), .isTRUE(local), .isTRUE(now))
        attr(handle, 'class') <- 'DLLHandle'
        
        name <- sub('\\.[[:alnum:]]+$', '', basename(x), FALSE, FALSE, FALSE, FALSE)

        init <- .Riposte('dynsym', handle, .pconcat('R_init_', name))
        if(!is.null(init)) {
            info <- .Riposte('dotC', init, list(info))[[1]]
        }
    }
    else {

        Call <- function(name, num) {
            x <- list(name=name,address=NULL,dll=NULL,numParameters=num)
            attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
            x
        }
        
        Fortran <- function(name, num) {
            x <- list(name=name,address=NULL,dll=NULL,numParameters=num)
            attr(x, 'class') <- c.default('FortranRoutine', 'NativeSymbolInfo')
            x
        }

        empty <- list()
        attr(empty, 'class') <- 'NativeRoutineList'
    
        call <- empty
        call[[1]] <- Call('R_addTaskCallback', 4)
        call[[2]] <- Call('R_getTaskCallbackNames', 0)
        call[[3]] <- Call('R_removeTaskCallback', 1)

        fortran <- empty
        fortran[[1]] <- Fortran('dqrcf', 8)
        fortran[[2]] <- Fortran('dqrdc2', 9)
        fortran[[3]] <- Fortran('dqrqty', 7)
        fortran[[4]] <- Fortran('dqrqy', 7)
        fortran[[5]] <- Fortran('dqrrsd', 7)
        fortran[[6]] <- Fortran('dqrxb', 7)
        fortran[[7]] <- Fortran('dtrco', 6)
    
        info[[1]] <- empty
        info[[2]] <- call
        info[[3]] <- fortran
        info[[4]] <- empty
    }
    
    routines <- info[1:4]
    attr(routines, 'useDynamicLookup') <- info[[5]]
    attr(routines, 'forceSymbols') <- info[[6]]
    attr(routines, 'handle') <- handle   
    attr(routines, 'class') <- 'DLLRegisteredRoutines'
 
    dllInfo <- list(name=name, path=x, dynamicLookup=FALSE, handle=handle, info=routines)
    attr(dllInfo, 'class') <- 'DLLInfo'
    loadedDLLs[[name]] <<- dllInfo
}

dyn.unload <- function(x) {
    .Riposte('dynunload', x)
    NULL
}

is.loaded <- function(symbol, PACKAGE, type) {
    .stop("is.loaded NYI")
}

getSymbolInfo <- function(symbol, PACKAGE, withRegistrationInfo) {
    
    if(!is.null(PACKAGE)) {
        p <- PACKAGE
        if(is.character(PACKAGE)) {
            if(PACKAGE == 'base')
                .stop("Can't look up symbol in the base package")

            p <- loadedDLLs[[PACKAGE]]
        
            if(is.null(p))
                .stop(sprintf("Package %s is not loaded", PACKAGE))
        
            p <- p$info
        }

        if(!inherits(p, 'DLLRegisteredRoutines', FALSE)) {
            .stop('getSymbolInfo needs a DLLRegisteredRoutines')
        }
    
        f <- .Riposte('dynsym', attr(p,'handle'), symbol)
    
        if(is.null(f))
            .stop(sprintf("no such symbol %s in package %s", symbol, PACKAGE))

    }
    else {
        for(p in loadedDLLs) {
            if(!is.null(p$handle)) {
                f <- .Riposte('dynsym', p$handle, symbol)
                if(!is.null(f))
                    break
            }
        }
        if(is.null(f))
            .stop(sprintf("no such symbol %s in any package", symbol))
    }
    
    attr(f, 'class') <- 'NativeSymbol'

    x <- list('name'=symbol, 'address'=f, 'package'=p)
    attr(x, 'class') <- c.default('FortranRoutine', 'NativeSymbolInfo')
    x
}

getRegisteredRoutines <- function(dllInfo) {
    dllInfo
}
