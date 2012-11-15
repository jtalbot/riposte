# adapted from the Intel ARBB example of the same name

# tests matrix indexing and updating (ideally without copying!)

kirchhoff_map <- function(traces, x, z, rvdt, dx, dz) {
	s <- dim(traces)
	x <- x*dx
	z <- z*dz

	i <- seq_len(s[1])
	cx <- i*x
	it <- sqrt((x-cx)*(x-cx) + z*z) * rvdt + 0.5
	sum(traces[i[it < s[2]], it[it < s[2]]])
}

kirchhoff_migration <- function(traces, velocity, dx, dt, dz) {
	model <- numeric(length(traces))
	dim(model) <- dim(traces)
	
	rvdt <- 1/(velocity*dt)

	for(x in 1:(dim(traces)[1])) {
		for(z in 1:(dim(traces)[2])) {
			model[x,z] <- kirchhoff_map(traces, x, z, rvdt, dx, dz)
		}
	}

	model
}

main <- function(width, height) {
	traces <- numeric(width*height)
	dim(traces) <- c(width, height)
	
	kirchhoff_migration(traces, 1, 1, 1, 1)
}

system.time(main(256, 256))
