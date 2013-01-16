
# 4.3.2 Argument matching

# Exact matching on tags

(f <- function (x, y)
x/y)

f(1, 2)
f(x=1, 2)
f(1, y=2)
f(x=1, y=2)
f(1, x=2)
f(y=1, 2)
f(y=1, x=2)

# error to have the same formal argument match several actuals or vice versa
#NYI: need a way to check that appropriate error messages are generated
#f(x=1, x=2)
#f(y=1, y=2)

#(g <- function(x, x)
#x / x)
#g(1, 2)
#g(x=1, 2)
#g(1, x=2)
#g(x=1, x=2)

#f(x=1, z=2)
#f(1, z=2)
