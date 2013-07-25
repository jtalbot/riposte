
# TODO: support localization

gettext <- function(domain, args) {
    args
}

ngettext <- function(n, msg1, msg2, domain) {
    if(n == 1L)
        msg1
    else
        msg2
}

bindtextdomain <- function(domain, dirname) {
    NULL
}

