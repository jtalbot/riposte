
.addCondHands <- function(name, handler, handle.env, other.env, run) {
    h <- handle.env[['.__handlers__.']]
    h[name] <- handler
    handle.env[['.__handlers__.']] <- h
}

.signalCondition <- function(cond, msg, call) {
    frame <- 2L
    cc <- class(cond)

    while(.frame(frame) != globalenv()
            && .env_has(.frame(frame), '.__parent__.')) {
        e <- .frame(frame)[['.__parent__.']]
        h <- e[['.__handlers__.']]
        n <- names(h)
        for(i in seq_len(length(n))) {
            if(any(cc == n[[i]])) {
                r <- list(cond, call, h[[i]])
                promise('throw', call('return', r), .frame(frame), .getenv(NULL))
                throw
            }
        }
        frame <- frame + 1L
    }
    NULL
}

.addRestart <- function(restart) {
    print(".addRestart called, but it's not implemented yet")
    NULL
}
