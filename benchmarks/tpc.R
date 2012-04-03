
# missing code so far

# data is in tests/tpc

# query definition is: 
# (from http://www.tpc.org/tpch/spec/tpch2.14.3.pdf)
#select
#l_returnflag, 
#l_linestatus, 
#sum(l_quantity) as sum_qty,
#sum(l_extendedprice) as sum_base_price,
#sum(l_extendedprice*(1-l_discount)) as sum_disc_price,
#sum(l_extendedprice*(1-l_discount)*(1+l_tax)) as sum_charge,
#avg(l_quantity) as avg_qty, 
#avg(l_extendedprice) as avg_price,
#avg(l_discount) as avg_disc, 
#count(*) as count_order
#from 
#lineitem
#where 
#l_shipdate <= date '1998-12-01' - interval '[DELTA]' day (3)
#group by 
#l_returnflag, 
#l_linestatus
#order by 
#l_returnflag, 
#l_linestatus;


format <- 
c(  NA,
	NA,
	NA,
	NA,
	"double", #quantity
	"double", #extendedprice
	"double", #discount
	"double", #tax
	"character", #returnflag 
	"character", #linestatus
	"date", #shipdate
	NA,
	NA,
	NA,
	NA,
	NA
)
r <- read.table("benchmarks/data/lineitem_small.tbl",sep="|",colClasses=format)
a <- ifelse(r[[5]] == 'A', 0L, ifelse(r[[5]] == 'N', 1L, 2L))
b <- ifelse(r[[6]] == 'F', 0L, 1L)
start_date <- 912499200-(90*24*60*60)
#f <- factor((a*2L+b)[r[[7]] <= start_date], 0L:5L)
f <- factor((a*2L+b), 0L:5L)

benchmark <- function() {
	z <- list(0)

	#filter <- r[[7]] <= start_date
	#z[[1]] <- lapply(split(r[[1]][filter], f), 'sum')
	#z[[2]] <- lapply(split(r[[2]][filter], f), 'sum')
	#z[[3]] <- lapply(split((r[[2]]*(1-r[[3]]))[filter], f), 'sum')
	#z[[4]] <- lapply(split((r[[2]]*(1-r[[3]])*(1+r[[4]]))[filter], f), 'sum')
	#z[[5]] <- lapply(split(r[[1]][filter], f), 'mean')
	#z[[6]] <- lapply(split(r[[2]][filter], f), 'mean')
	#z[[7]] <- lapply(split(r[[3]][filter], f), 'mean')
	#z[[8]] <- lapply(split(r[[1]][filter], f), 'length')

	z[[1]] <- lapply(split(r[[1]], f), 'sum')
	z[[2]] <- lapply(split(r[[2]], f), 'sum')
	z[[3]] <- sum(r[[3]])
	z[[4]] <- sum(r[[4]])
	#z[[1]] <- lapply(split(r[[1]], f), 'sum')
	#z[[2]] <- lapply(split(r[[2]], f), 'sum')
	#z[[3]] <- lapply(split((r[[2]]*(1-r[[3]])), f), 'sum')
	#z[[4]] <- lapply(split((r[[2]]*(1-r[[3]])*(1+r[[4]])), f), 'sum')
	#z[[5]] <- lapply(split(r[[1]], f), 'mean')
	#z[[6]] <- lapply(split(r[[2]], f), 'mean')
	#z[[7]] <- lapply(split(r[[3]], f), 'mean')
	#z[[8]] <- lapply(split(r[[1]], f), 'length')
	z
}

system.time(for(i in 1:100) force(benchmark()))
