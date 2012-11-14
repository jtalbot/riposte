simpsonsAux <- function(f, a, b, eps, S, fa, fb, fc, bottom) {

    c <- (a+b)/2
    h <- b-a

    d <- (a+c)/2
    e <- (c+b)/2

    fd <- f(d)
    fe <- f(e)

    Sl <- (h/12)*(fa+4*fd+fc)
    Sr <- (h/12)*(fc+4*fe+fb)

    S2 <- Sl+Sr

    if(abs(S2-S) <= 15*eps)
        S2+(S2-S)/15
    else
        simpsonsAux(f, a, c, eps/2, Sl, fa, fc, fd, bottom-1) +
        simpsonsAux(f, c, b, eps/2, Sr, fc, fb, fe, bottom-1)
}

simpsons <- function(f, a, b, eps, bottom) {
    c <- (a+b)/2
    h <- b-a
    fa <- f(a)
    fb <- f(b)
    fc <- f(c)
    S <- (h/6)*(fa+4*fc+fb)
    simpsonsAux(f, a, b, eps, S, fa, fb, fc, bottom)
}

simpsons(sin, 0, 100, 0.0000000000001, 10000)
system.time(simpsons(sin, 0, 100, 0.0000000000001, 10000))

