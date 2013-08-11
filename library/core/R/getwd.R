
getwd <- function()
    .External('getwd_impl')

setwd <- function(dir) {
    dir <- as.character.default(dir)
    if(length(dir) != 1L)
        .stop("setwd argument must be of length 1")
    .External('setwd_impl', dir)
}

