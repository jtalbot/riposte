
cumsum <- function(x) UseGroupMethod('cumsum', 'Math', x)
cumsum.default <- function(x) cumsum(strip(x))

cumprod <- function(x) UseGroupMethod('cumprod', 'Math', x)
cumprod.default <- function(x) cumprod(strip(x))

cummax <- function(x) UseGroupMethod('cummax', 'Math', x)
cummax.default <- function(x) cummax(strip(x))

cummin <- function(x) UseGroupMethod('cummin', 'Math', x)
cummin.default <- function(x) cummin(strip(x))

