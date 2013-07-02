
lapply <- function(x, func) {
	.External(mapply(list(x), func))
}

