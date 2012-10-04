# doubles
c(0)!=c(0)
c(1,2)!=c(1,2)
c(3,4)!=c(2,5)
c(4,5,6)!=c(10,5,20)

# integers 
c(0L)!=c(0L)
c(1L,2L)!=c(1L,2L)
c(3L,4L)!=c(2L,5L)
c(4L,5L,6L)!=c(10L,5L,20L)

#complex
c(0i)!=c(0i)
c(0i,1i)!=c(1i,0i)
c(0+0i,0+1i,1+1i)!=c(0+0i,1+0i,2+3i)

#characters
c("") != c("a")
c("a","bat") != c("a","cries")
c("a","bat","cries") != c("\n","a","bat")

#logicals
c(FALSE)!=c(FALSE)
c(FALSE,TRUE) != c(TRUE,TRUE)
c(FALSE,TRUE,TRUE)!=c(TRUE,FALSE,TRUE)
