
!c(5L,6L)
!c(0L,10L)
!c(10,20)
!c(0,5)

c(FALSE,TRUE) | c(0L,2L)
c(0,1) | c(FALSE,FALSE)
c(2,4,6) | c(5L,4L,3L)
c(0,0,0,0) | c(0L,0L,0L,0L)

c(FALSE,TRUE) & c(0L,1L)
c(0,10) & c(FALSE,TRUE)
c(2,4,6) & c(5L,6L,7L)
c(0) & c(0L)

c(FALSE,TRUE) == c(0L,1L)
c(5L,7L) == c(TRUE,FALSE)
c(5,10) == c(5L,7L)
c(5L,15L) == c(6,10)

c(FALSE,TRUE) != c(0L,1L)
c(5L,15L) != c(TRUE,FALSE)
c(5,0) != c(5L,10L)
c(5L,4L) != c(6,10)

c(FALSE,TRUE) <= c(0L,1L)
c(5L,9L) <= c(TRUE,FALSE)
c(5,6) <= c(5L,0L)
c(5L,6L) <= c(6,5)

c(FALSE,TRUE) > c(0L,10L)
c(5L,3L) > c(TRUE,FALSE)
c(5,4) > c(5L,10L)
c(5L,3L) > c(6,10)
