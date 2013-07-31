
Sys.setenv <- function(names, values) {
    .Map('syssetenv_map', list(names, values), 'logical')[[1]]
}

Sys.unsetenv <- function(x) {
    .Map('sysunsetenv_map', list(x), 'logical')[[1]]
}
