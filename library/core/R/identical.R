
identical <- function(x, y) {
       (typeof(x) == typeof(y)) && 
       all( ((!is.na(x) && !is.na(y) && (x == y))) ||
            (is.na(x) && is.na(y)))
}
