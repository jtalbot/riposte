
split <- function(x, f) {
	split(strip(x), strip(f), length(attr(f, 'levels')))
}

