
.addCondHands <- function(name, handler, handle.env, other.env, run) {
    h <- handle.env[['__handlers__']]
    h[name] <- handler
    handle.env[['__handlers__']] <- h
}

.signalCondition <- function(cond, msg, call) {
    frame <- 2L
    cc <- class(cond)

    while(.frame(frame)[[1L]] != globalenv()
            && !is.null(.frame(frame)[[6L]])) {
        e <- .frame(frame)[[1L]]
        h <- e[['__handlers__']]
        n <- names(h)
        for(i in seq_len(length(n))) {
            if(any(cc == n[[i]])) {
                r <- list(cond, call, h[[i]])
                promise('throw', call('return', r), e, .getenv(NULL))
                throw
            }
        }
        frame <- frame + 1L
    }
    NULL
}

