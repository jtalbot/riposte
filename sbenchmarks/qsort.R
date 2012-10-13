
# from Julia's benchmarks

N <- as.integer(commandArgs(TRUE)[[1]])

qsort = function(a) {
    qsort_kernel = function(lo, hi) {
        i = lo
        j = hi
        while (i < hi) {
            pivot = a[floor((lo+hi)/2)]
            while (i <= j) {
                while (a[i] < pivot) i = i + 1
                while (a[j] > pivot) j = j - 1
                if (i <= j) {
                    t = a[i]
                    a[i] <<- a[j]
                    a[j] <<- t
                    i = i + 1;
                    j = j - 1;
                }
            }
            if (lo < j) qsort_kernel(lo, j)
            lo = i
            j = hi
        }
    }
    qsort_kernel(1, length(a))
    return(a)
}

sortperf = function(n) {
    v = runif(n)
    return(qsort(v))
}

cat(system.time(sortperf(N))[[3]])
