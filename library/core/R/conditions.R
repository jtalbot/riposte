
signalCondition <- function(cond, call) {
    frame <- 1L
    cc <- class(cond)
    
    while(parent.frame(frame) != globalenv()) {
        e <- parent.frame(frame)
        h <- e[['__handlers__']]
        n <- names(h)
        for(i in seq_len(length(n))) {
            if(any(cc == n[[i]])) {
                r <- list(cond, call, h[[i]])
                promise('throw', call('return', r), e, parent.frame(0))
                throw
            }
        }
        frame <- frame + 1L
    }
    NULL
}
