
c <- function(...) unlist(list(...))

environment <- function(fun=NULL) .Internal(environment)(fun)

ifelse <- function(test, yes, no) {
	if(!any(test)) no
	else {
		tmp <- yes
		if(!all(test)) yes[!test] <- no
		tmp
	}
}
