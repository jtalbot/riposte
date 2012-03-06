
runif <- function(n, min=0, max=1) {
	if(missing(min) && missing(max))
		random(n)
	else
		random(n)*(max-min)+min
}
