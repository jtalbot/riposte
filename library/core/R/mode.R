
`storage.mode<-` <- function(x, value) {
	switch(value,
		logical=as(x,"logical"), 
		integer=as(x,"integer"),
		double=as(x,"double"),
		complex=as.complex(x),
		raw=as(x,"raw"),
		character=as(x,"character"),
		list=as(x,"list"),
		expression=as.expression(x),
		name=as.name(x),
		symbol=as.symbol(x), 
		.stop("unsupported mode"))
}

