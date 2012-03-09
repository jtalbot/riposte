
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
r <- read.table("data/lineitem.tbl","|",fmt)
a <- ifelse(r[[5]] == 'A', 0, ifelse(r[[5]] == 'N', 1, 2))
b <- ifelse(r[[6]] == 'F', 0, 1)
f <- factor(a*2+b, 0:5)
start_date <- 912499200

benchmark <- function() {
	sum_qty <- lapply(split(r[[1]][r[[7]]<=start_date], f), 'sum')
	sum_base_price <- lapply(split(r[[2]][r[[7]]<=start_date], f), 'sum')
	sum_disc_price <- lapply(split((r[[2]]*(1-r[[3]]))[r[[7]]<=start_date], f), 'sum')
	sum_charge <- lapply(split((r[[2]]*(1-r[[3]])*(1+r[[4]]))[r[[7]]<=start_date], f), 'sum')
	avg_qty <- lapply(split(r[[1]][r[[7]]<=start_date], f), 'mean')
	avg_price <- lapply(split(r[[2]][r[[7]]<=start_date], f), 'mean')
	avg_disc <- lapply(split(r[[3]][r[[7]]<=start_date], f), 'mean')
	count_order <- lapply(split(r[[7]][r[[7]]<=start_date], f), 'length')

	cat(sum_qty,"\n")
	cat(sum_base_price,"\n")
	cat(sum_disc_price,"\n")
	cat(sum_charge,"\n")
	cat(avg_qty,"\n")
	cat(avg_price,"\n")
	cat(avg_disc,"\n")
	cat(count_order,"\n")
}

benchmark()
