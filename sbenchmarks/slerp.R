slerp <- function(p0, p1, t) {
	return (1-t)*p0 + t*p1
}
slerp(10, 11, .7)
