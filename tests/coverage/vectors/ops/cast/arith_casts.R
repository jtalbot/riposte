# tests implicit casting due to arithmetic operations

-c(TRUE,FALSE)
+c(FALSE,FALSE,TRUE)

-c(5L,6L)
+c(10L,0L,5L)

c(1L,2L)+c(1,5)
c(5,10)+c(10L,20L)

c(TRUE,FALSE)+c(1L,10L)
c(1L,2L)+c(FALSE,FALSE)

c(FALSE,TRUE)+c(2,3)
c(2,4)+c(TRUE,TRUE)

c(TRUE,TRUE)*c(TRUE,FALSE)
c(TRUE,FALSE)*c(FALSE,TRUE)

c(3L,4L)^c(2L,3L)
c(2L,5L)^c(1L,0L)

c(3L,5L)/c(2L,5L)
c(3L,3L)%/%c(2L,1L)

c(FALSE,FALSE)/c(TRUE,FALSE)
c(FALSE,TRUE)%/%c(TRUE,TRUE)

c(FALSE,TRUE)/c(5L,2L)
c(TRUE,TRUE)%/%c(10L,6L)

c(5,6)/c(TRUE,FALSE)
c(5,10)%/%c(TRUE,TRUE)

c((1+2i),(2+4i))+c(1,2)
c(4L,6L)+c((1+2i),(2+4i))
c((1+2i),(2+4i))+c(TRUE,FALSE)

