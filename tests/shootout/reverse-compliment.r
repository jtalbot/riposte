# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org
# contributed by Saptarshi Guha and Justin Talbot

wrap <- function(str, width=60)
{
  for(i in seq.int(1,nchar(str),by=width)){
    cat(substr(str,i,(i+width-1)), "\n", sep="")
  }
}

stdin <- file("stdin")
lines <- readLines(stdin)
close(stdin)

sections <- c(grep("^>", lines), length(lines)+1)

for(i in 1:(length(sections)-1))
{
  cat(lines[sections[i]], "\n", sep="")
  txt <- paste(lines[(sections[i]+1):(sections[i+1]-1)], collapse="")
  txt <- paste(rev(strsplit(txt, split="")[[1]]), collapse="")
  wrap( chartr('ACBDGHK\nMNSRUTWVYacbdghkmnsrutwvy',
               'TGVHCDM\nKNSYAAWBRTGVHCDMKNSYAAWBR', txt) )
}
