
tabulate <- function(bin, nbins) {
    .Riposte('tabulate', bin[!is.na(bin) & bin>=1L & bin <= nbins], nbins)
}

