
writeLines <- function(text, con, sep, useBytes) 
    .writeLines(con, text, sep, useBytes)

.writeLines <- function(con, text, sep, useBytes)
    UseMethod('.writeLines', con)

.writeLines.file <- function(con, text, sep, useBytes) {
    .open.file(con, 'wt')
    .Riposte('file_writeLines', attr(con, 'conn_id'), as.character(text), as.character(sep))
    NULL
}

.writeLines.gzfile <- function(con, text, sep, useBytes) {
    .stop("writelines.gzfile is not yet implemented")
    NULL
}

.writeLines.terminal <- function(con, text, sep, useBytes) {
    .Riposte('terminal_writeLines', strip(con), as.character(text), as.character(sep))
    NULL
}

