
deparse(1)

deparse(c(1,2,3,4,5))
(a <- c(1,2,3,4,5))
deparse(a)

#DIFF: R's deparsing deparses this to "1:5", Riposte deparses to "c(1,2,3,4,5)"
#deparse(1:5)
#
#a <- 1:5
#deparse(a)
