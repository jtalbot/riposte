fankuch <- function(n){
   p <- 1:n; q <- 1:n; s <- 1:n; sign <- 1; maxflips <- 0;sum <- 0
   while(TRUE){
     q1 <- p[1]				
     if(q1 != 1){
       q[2:n] <- p[2:n]
       flips <- 1
       while(TRUE){
	qq <- q[q1]
	if(qq == 1){
	  sum = sum + sign*flips
	  maxflips <- max(maxflips,flips )
	  break
        }
	q[q1] <- q1
	if(q1 >= 4){
	  i <- 2; j <- q1 - 1
	  repeat{
            t <- q[i]
            q[i] <- q[j]
            q[j] = t
            i <- i + 1; j <- j - 1;
            if(i >= j) break
          }
	}
	q1 <- qq;
        flips <- flips + 1
      }
     }
    if(sign == 1){
      t <- p[2]
      p[2] <- p[1] ; p[1] <- t
      sign <- -1	
    }else{
      t <- p[2]
      p[2] <- p[3] ;p[3] <-t;
      sign <- 1
      for(i in 3:n){
	sx <- s[i]
	if(sx != 1) {
          s[i] = sx-1;
          break
        }
	if( i == n) return(c(sum, maxflips))
	s[i] = i
	t = p[1];
        p[1:i] <- p[2:(i+1)]
        p[i+1] <- t
      }
    }
   }
 }

n <- 9 #as.numeric(commandArgs(TRUE)[1])
p <- fankuch(n)
cat(paste(p[1], "\nPfannkuchen(", n, ") = ", p[2], "\n"))
