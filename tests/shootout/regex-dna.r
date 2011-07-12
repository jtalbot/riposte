# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org
# contributed by Saptarshi Guha and Justin Talbot

stdin <- file("stdin")
data <- readLines(stdin)
close(stdin)

data <- paste(c(data, ""), collapse="\n") # add newlines back in
l1 <- nchar(data)

data <- gsub(">.*?\n|\n", "", data)
l2 <- nchar(data)

patterns <- c(
  "agggtaaa|tttaccct",
  "[cgt]gggtaaa|tttaccc[acg]",
  "a[act]ggtaaa|tttacc[agt]t",
  "ag[act]gtaaa|tttac[agt]ct",
  "agg[act]taaa|ttta[agt]cct",
  "aggg[acg]aaa|ttt[cgt]ccct",
  "agggt[cgt]aa|tt[acg]accct",
  "agggta[cgt]a|t[acg]taccct",
  "agggtaa[cgt]|[acg]ttaccct")

for(p in patterns){
  count <- sum(gregexpr(p,data)[[1]]>=0)
  cat(sprintf("%s %s\n",p,count))
}

y <- c(
  B="(c|g|t)", D="(a|g|t)",   H="(a|c|t)", K="(g|t)",
  M="(a|c)",   N="(a|c|g|t)", R="(a|g)",   S="(c|g)",
  V="(a|c|g)", W="(a|t)",     Y="(c|t)")
  
for(i in seq_along(y)){
  data <- gsub(names(y)[i], y[i], data, perl=TRUE)
}

cat(sprintf("\n%s\n%s\n%s\n", l1, l2, nchar(data)))
