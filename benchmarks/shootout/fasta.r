# The Computer Language Benchmarks Game
# http://shootout.alioth.debian.org
# contributed by Justin Talbot and Saptarshi Guha

lcg.last <- 42L
lcg <- function(n) {
#   IM <- 139968L,  IA <- 3877L,  IC <- 29573L
    r <- integer(n) 
	for(i in seq_along(r)) {
		r[i] <- lcg.last <- (lcg.last * 3877L + 29573L) %% 139968L 
	}
    lcg.last <<- lcg.last
	r / 139968L 
}

wrap <- function(str, width=60)
{
  for(i in seq.int(1,nchar(str),by=width)){
    cat(substr(str,i,(i+width-1)), "\n", sep="")
  }
}

ALU <- strsplit(paste(
	"GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGG",
    "GAGGCCGAGGCGGGCGGATCACCTGAGGTCAGGAGTTCGAGA",
    "CCAGCCTGGCCAACATGGTGAAACCCCGTCTCTACTAAAAAT",
    "ACAAAAATTAGCCGGGCGTGGTGGCGCGCGCCTGTAATCCCA",
    "GCTACTCGGGAGGCTGAGGCAGGAGAATCGCTTGAACCCGGG",
    "AGGCGGAGGTTGCAGTGAGCCGAGATCGCGCCACTGCACTCC",
    "AGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA", sep=""), split="")[[1]]
    
makeRepeatFasta <- function(id, desc, chars, n)
{
	cat(">", id, " ", desc, "\n", sep="")
    txt <- chars[rep(1:length(chars), length.out=n)]
	wrap(paste(txt, collapse=""))
}


IUB.c <- c('a', 'c', 'g', 't', 'B', 'D', 'H', 'K', 
		'M', 'N', 'R', 'S', 'V', 'W', 'Y')
IUB.p <- c(0.27, 0.12, 0.12, 0.27, rep(0.02,11))

Homo.Sapiens.c <- c('a', 'c', 'g', 't')
Homo.Sapiens.p <- c(0.3029549426680,
				0.1979883004921,
				0.1975473066391,
				0.3015094502008)
				
makeRandomFasta <- function(id, desc, chars, probs, n)
{
	cat(">", id, " ", desc, "\n", sep="")
    txt <- chars[findInterval(lcg(n), cumsum(probs))+1]
	wrap(paste(txt, collapse=""))
}


N <- as.numeric(commandArgs(TRUE)[1])
makeRepeatFasta("ONE", "Homo sapiens alu", ALU, N * 2)
makeRandomFasta("TWO", "IUB ambiguity codes", IUB.c, IUB.p, N * 3)
makeRandomFasta("THREE", "Homo sapiens frequency", Homo.Sapiens.c, Homo.Sapiens.p, N * 5)

