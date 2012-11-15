p <- read.table("data/lr_p.txt")[[1]]
dim(p) <- c(length(p)/30, 30)
r <- read.table("data/lr_r.txt")[[1]]

system.time(glm(r~p-1, family=binomial(link="logit")))
