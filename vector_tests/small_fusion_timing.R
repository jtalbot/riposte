#trace.config(2)
simple <- function() {
	a <- 0
	for(i in 1:N_TIMES) {
		a <- a + max(v + v + v)
	}
}

for(i in 7:20) {
	WIDTH <- 2 ** i
	v <- 1 : WIDTH
	N_TIMES <-  60 * (2 ** 20) / WIDTH
	time <- system.time(simple())
	cat(WIDTH)
	cat("\t")
	cat(time / N_TIMES)
	cat("\n")
}