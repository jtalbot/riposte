# Goal: nowhere close to this yet
# auto generate valid R programs
# user can parametrically select language features enabled

Double <- function(opts, length=rpois(1,1), digits=rpois(length,1)) {
    if(length > 0)
        paste(round(rnorm(length,0,3), digits))
    else
        ""
}

Integer <- function(opts, length=rpois(1,1)) {
    if(length > 0)
        paste(as.integer(runif(length,-10,10)),"L",sep="")
    else
        ""
}

Complex <- function(opts, length=rpois(1,1), digits=rpois(length,1)) {
    if(length > 0)
        paste(round(rnorm(length,0,3), digits),'i',sep="")
    else 
        ""
}

Logical <- function(opts, length=rpois(1,1)) {
    paste(rnorm(length)<0)
}

Null <- function(opts, length=1) {
    "NULL"
}

Character <- function(opts, length=rpois(1,1)) {
    paste('"',letters[sample(26,length)],'"',sep="")
}

Vector <- function(opts, types, length=rpois(1,1)) {

    v = paste(sample(types,1)[[1]](opts, length),collapse=",")

    if(length != 1 || runif(1) < 0.5) {
        paste("c(",v,")")
    }
    else {
        v
    }
}

DoubleVector <- function(opts, length) Vector(opts, c(Double), length)
IntegerVector <- function(opts, length) Vector(opts, c(Integer), length)
ComplexVector <- function(opts, length) Vector(opts, c(Complex), length)
LogicalVector <- function(opts, length) Vector(opts, c(Logical), length)
NullVector <- function(opts, length) Vector(opts, c(Null), length)
CharacterVector <- function(opts, length) Vector(opts, c(Character), length)

LogicalExpression <- function(opts, length) {
    f <- c( LogicalVector )
    if(opts$coerce) {
        f <- c(f, DoubleVector, IntegerVector, ComplexVector)
        if(!missing(length) && length == 0)
            f <- c(f, NullVector)

        if(opts$na)
            f <- c(f, CharacterVector)
    }

    if(opts$logicalops)
        f <- c(f, LogicalOp)

    sample(f, 1)[[1]](opts, length)
}
