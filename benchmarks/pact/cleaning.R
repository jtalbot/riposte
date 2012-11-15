
data <- as.double(1:20000000)
force(data)

bench <- function() {

	z.score <- function(data, m=mean(data), stdev=sd(data)) {
		# these two lines force the promises, allowing us to fuse m and stdev
		# otherwise they are separated by the barrier (data-m), and don't fuse.
		# can we do better?
		m
		stdev
		(data-m) / stdev
	}

	outliers <- function(data, ignore) {
		use <- !ignore(data)
		z <- z.score(data, mean(data[use]), sd(data[use]))
		sum(abs(z) > 1)
	}

	outliers(data, function(x) { is.na(x) | x==9999 })
}

#bench()
system.time(bench())
