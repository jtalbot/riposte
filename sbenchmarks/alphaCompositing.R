alphaCompositing <- function(color1, color2, alpha1, alpha2) {
	color0 <- color1*alpha1 + color2*alpha2*(1-alpha1)
	return(color0)
}

color1 <- 1:10
color2 <- 1:10
alpha1 <- 0.4
alpha2 <- 0.7

a=0;
run <- function() {
    b = color1
    while(a<10000000) {
        a=a+1;
        b = alphaCompositing(b, color2, alpha1, alpha2)
    }
    b
}

system.time(run())
