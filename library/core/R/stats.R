
var <- function(x) {
	cm2(x,x)/(length(x)-1)
}

sd <- function(x) {
	sqrt(var(x))
}

cov <- function(x,y) {
	cm2(x,y)/(length(x)-1)
}

