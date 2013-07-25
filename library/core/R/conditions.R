
.addCondHands <- function(name, handler, eval.env, handle.env, run) {
    h <- handle.env[['__handlers__']]
    h[name] <- handler
    handle.env[['__handlers__']] <- h
}
