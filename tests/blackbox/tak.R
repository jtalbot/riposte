(tak <- function (x, y, z) 
{
	if (y >= x) { 
        z
    }
	else {
        a <- tak(x - 1, y, z)
        b <- tak(y - 1, z, x)
        c <- tak(z - 1, x, y)
        tak(a, b, c)
    }
})

tak(24, 16, 8)
