
.subset <- function(x, ...) `[`(unclass(x), ...)

.subset2 <- function(x, ...) `[[`(unclass(x), ...)

.isMethodsDispatchOn <- function(onOff) {
    FALSE
}

