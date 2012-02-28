
outliers <- function(data, ignore) {
	x <- data[ignore(data)]
	m <- sum(x)/length(x)
	sd <- sqrt(sum(x^2)/length(x))
	z <- (data-m)/sd
	abs(z) > 3
}

outliers(data, function(x) { is.na(x) || x==999 })

