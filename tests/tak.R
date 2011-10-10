tak <- function(x, y, z) {
	if(y >= x) z
	else tak( tak(x-1, y, z), tak(y-1, z, x), tak(z-1, x, y) )
}

system.time(tak(24, 16, 8))
