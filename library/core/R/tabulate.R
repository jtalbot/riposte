
tabulate <- function(bin, nbins) {
    .External('tabulate', bin[!is.na(bin) & bin>=1L & bin <= nbins], nbins)
}

