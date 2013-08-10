
mapply <- function(FUN, dots, MoreArgs) {
	.External('mapply', dots, FUN)
}

