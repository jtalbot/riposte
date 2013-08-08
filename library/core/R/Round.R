
floor <- function(x) UseGroupMethod('floor', 'Math', x)
floor.default <- function(x) .ArithUnary2('floor_map', x)

ceiling <- function(x) UseGroupMethod('ceiling', 'Math', x)
ceiling.default <- function(x) .ArithUnary2('ceiling_map', x)

trunc <- function(x, ...) UseGroupMethod('trunc', 'Math', x)
trunc.default <- function(x) .ArithUnary2('trunc_map', x)

round <- function(x, digits = 0) UseGroupMethod('round', 'Math', x)
round.default <- function(x, digits = 0) .Digits('round_map', x, digits)

signif <- function(x, digits = 6) UseGroupMethod('signif', 'Math', x)
signif.default <- function(x, digits = 6) .Digits('signif_map', x, digits)

