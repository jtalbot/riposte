tak <- function(x, y, z) {
	if(y >= x) z
	else tak( tak(x-1, y, z), tak(y-1, z, x), tak(z-1, x, y) )
}

for(i in 1:10) tak(24, 16, 8)
