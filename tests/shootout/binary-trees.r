# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org
# contributed by Saptarshi Guha and Justin Talbot
# modified from Mike Pall's Lua implementation

bottomUpTree <- function(item, depth)
{
  if(depth > 0)
    list(item, bottomUpTree(item+item-1, depth-1), bottomUpTree(item+item, depth-1))
  else
    list(item)
}

itemCheck <- function(tree)
{
  tree[[1]] + if(length(tree) == 3) itemCheck(tree[[2]]) - itemCheck(tree[[3]]) else 0
}

#N <- as.numeric(commandArgs(TRUE)[1])
N <- 14 

mindepth <- 4
maxdepth <- max(mindepth+2, N)

cat(paste("stretch tree of depth ", maxdepth+1, 
  "\t check: ", itemCheck(bottomUpTree(0, maxdepth+1)), "\n", sep=""))

longLivedTree <- bottomUpTree(0, maxdepth)

#for(depth in seq.int(mindepth, maxdepth, 2))
for(depth in seq(mindepth, 2, (maxdepth-mindepth)/2+1))
{
  iterations <- 2^(maxdepth-depth+mindepth)
  check <- 0
  for(i in 1:iterations)
    check <- check + itemCheck(bottomUpTree(1, depth)) + itemCheck(bottomUpTree(-1, depth))
  cat(paste(iterations*2, "\t trees of depth ", depth, 
    "\t check: ", check, "\n", sep=""))
}

cat(paste("long lived tree of depth ", maxdepth, 
  "\t check: ", itemCheck(longLivedTree), "\n", sep=""))

