options(echo = FALSE)

fannkuch <- function(n)
{
    p <- 1L:n
    q <- 1L:n
    s <- 1L:n
    sign <- 1
    maxflips <- 0
    sum <- 0

    while(TRUE)
    {
        q1 <- p[1L]    
        if(q1 != 1L)
        {
            q[2L:n] <- p[2L:n]
            flips <- 1
            while(TRUE)
            {
                qq <- q[q1]
                if(qq == 1)
                {
                    sum = sum + sign*flips
                    maxflips <- pmax(maxflips,flips)
                    break
                }
                q[q1] <- q1
                if(q1 >= 4L)
                {
                    i <- 2L; j <- q1 - 1L
                    repeat
                    {
                        t <- q[i]
                        q[i] <- q[j]
                        q[j] = t
                        i <- i + 1L; j <- j - 1L;
                        if(i >= j) break
                    }
                }
                q1 <- qq
                flips <- flips + 1
            }
        }

        if(sign == 1)
        {
            t <- p[2L]
            p[2L] <- p[1L] ; p[1L] <- t
            sign <- -1    
        }
        else
        {
            t <- p[2L]
            p[2L] <- p[3L] ;p[3L] <-t;
            sign <- 1
            for(i in 3L:n)
            {
                sx <- s[i]
                if(sx != 1L)
                {
                    s[i] <- sx-1L
                    break
                }
                if(i == n) 
                    return(c(sum, maxflips))
                s[i] = i
                t = p[1L]
                p[1L:i] <- p[2L:(i+1L)]
                p[i+1L] <- t
            }
        }
    }
}

run <- function()
{
    n <- as.integer(commandArgs(TRUE)[1L])
    p <- fannkuch(n)
    cat(p[1], "\n")
    cat("Pfannkuchen(", n, ") = ", p[2L], "\n", sep="")
}

system.time(run())
