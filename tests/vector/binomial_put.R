#adapted from https://github.com/ispc/ispc/tree/master/examples/options
N_OPTIONS <- 65536
BINOMIAL_NUM <- 64
S <- rep(100,1,N_OPTIONS)
X <- rep(98,1,N_OPTIONS)
T <- rep(2, 1,N_OPTIONS)
r <- rep(.02,1,N_OPTIONS)
v <- rep(5,1,N_OPTIONS)

i <- 1:N_OPTIONS
dt <- T / BINOMIAL_NUM
u <- exp(v * sqrt(dt))
d <- 1.0 / u
disc <- exp(r * dt)
Pu <- (disc - d) / (u - d)

j <- rep(1:BINOMIAL_NUM,N_OPTIONS)
upow <- u ** (2 * (j - 1) - BINOMIAL_NUM)
V <- pmax(0.0, X - S * upow)

V0 <- (0: (N_OPTIONS - 1)) * BINOMIAL_NUM
for(jj in (BINOMIAL_NUM - 1) : 1) {
	for(k in 0 : (jj - 1) ) {
		idx <- V0 + (k + 1)
		V[idx] <- ((1 - Pu) * V[idx] + Pu * V[idx + 1]) / disc
	}
}
result <- sum(V[1 + V0]) / N_OPTIONS
