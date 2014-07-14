
loadedDLLs <- vector('list',0) 
attr(loadedDLLs, 'class') <- 'DLLInfoList'

getLoadedDLLs <- function() {
    loadedDLLs
}

dyn.load <- function(x, local, now, str) {
    if(x != 'base') {
        p <- .External('dynload', as.character(x), .isTRUE(local), .isTRUE(now))
        attr(p, 'class') <- 'DLLHandle'

        name <- basename(x)

        a <- list(name=name, path=x, dynamicLookup=FALSE, handle=p, info=NULL)
    }
    else {
        a <- list(name=x, path=x, dynamicLookup=FALSE, handle=NULL, info=NULL)
    }
    attr(a, 'class') <- 'DLLInfo'
    loadedDLLs[[x]] <<- a
}

dyn.unload <- function(x) {
    .External('dynunload', x)
    NULL
}

is.loaded <- function(symbol, PACKAGE, type) {
    .stop("is.loaded NYI")
}

getRegisteredRoutines <- function(dllInfo) {
    r <- list()

    a <- list()
    attr(a, 'class') <- 'NativeRoutineList'
    
    call <- list()

    x <- list(name='R_addTaskCallback',address=NULL,dll=NULL,numParameters=4)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    call[[1]] <- x

    x <- list(name='R_getTaskCallbackNames',address=NULL,dll=NULL,numParameters=0)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    call[[2]] <- x
    
    x <- list(name='R_removeTaskCallback',address=NULL,dll=NULL,numParameters=1)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    call[[3]] <- x
    
    attr(call, 'class') <- 'NativeRoutineList'


    fortran <- list()

    x <- list(name='dqrcf',address=NULL,dll=NULL,numParameters=8)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[1]] <- x
 
    x <- list(name='dqrdc2',address=NULL,dll=NULL,numParameters=9)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[2]] <- x
    
    x <- list(name='dqrqty',address=NULL,dll=NULL,numParameters=7)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[3]] <- x
    
    x <- list(name='dqrqy',address=NULL,dll=NULL,numParameters=7)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[4]] <- x
    
    x <- list(name='dqrrsd',address=NULL,dll=NULL,numParameters=7)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[5]] <- x
    
    x <- list(name='dqrxb',address=NULL,dll=NULL,numParameters=7)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[6]] <- x
    
    x <- list(name='dtrco',address=NULL,dll=NULL,numParameters=6)
    attr(x, 'class') <- c.default('CallRoutine', 'NativeSymbolInfo')
    fortran[[7]] <- x
    
    r$.C <- a
    r$.Call <- call
    r$.Fortran <- fortran
    r$.External <- a

    r
}
