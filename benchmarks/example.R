data <- list(
	as.double(1:20000000),
	as.double(1:20000000)
)

bin <- function(x) { ifelse(x > 0, 1, ifelse(x < 0, -1, 0)) }
ignore <- function(x) { is.na(x) | x == 9999 }

clean <- function(data) {
	data[[2]][!ignore(data[[1]]) & bin(data[[1]]) == 1]
}

mean(clean(data))
