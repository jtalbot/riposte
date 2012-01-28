fail <- 0
testno <- 0
{
	cat("Testing if statement\n-------------\n")
	if(TRUE) cat("PASS\n") else cat ("FAIL\n")
	if(FALSE) cat("FAIL\n") else cat ("PASS\n")
	if(1) cat("PASS\n") else cat("FAIL\n")
	if(is.null(if(FALSE) 1)) cat("PASS\n") else cat("FAIL\n")
	"done"
}

{
	PassIfTrue <- function(x) { fail<-fail+1; testno <<- testno + 1; cat(testno-1); if(x) { cat(" PASS\n"); fail<-fail-1 } else cat(" FAIL\n") }
	PassIfFalse <- function(x) { fail<-fail+1;testno <<- testno + 1; cat(testno-1); if(x) cat(" FAIL\n") else { cat(" PASS\n"); fail<-fail-1 } }
	PassIfEq <- function(x,y) { testno <<- testno + 1; cat(testno-1); if(x == y) cat(" PASS\n") else { cat(" FAIL:\nx: ");  cat(x); cat("\ny: "); cat(y); cat("\n"); fail<<-fail+1; } }
	foo <- function(x,y) { x + y; }	
}

{
	trace.config(0)
	d <- 1:128
	s <- 1:65
	i <- as.integer(d)
	l <- (d < 50) 
	ds <- 3
	is <- 3L
	ls <- FALSE
	
	r0 <- d + d
	r1 <- d + ds
	r2 <- ds + d
	r3 <- ds + ds
	r4 <- i + i
	
	r5 <- i + is
	r6 <- is + i
	r7 <- is + is
	r8 <- d + 3
	r9 <- 3 + d
	r10 <- 3 + 3
	r11 <- i + 3L
	r12 <- 3L + i
	r13 <- 3L + 3L
	
	r14 <- d + i
	r15 <- i + d
	r16 <- d + is
	r17 <- d + 3L
	r18 <- is + d
	r19 <- 3L + d
	r20 <- i + ds
	r21 <- ds + i
	r22 <- 3 + i
	r23 <- i + 3
	
	r24 <- l + d
	r25 <- d + l
	r26 <- i + l
	r27 <- l + i
	r28 <- ls + d
	r29 <- d + ls
	r30 <- i + ls
	r31 <- ls + i
	
	r32 <- l + ds
	r33 <- ds + l
	r34 <- is + l
	r35 <- l + is
	
	r36 <- -d
	r37 <- -i
	r38 <- -ds
	r39 <- -is
	
	r40 <- d + d + d + d
	r41 <- d + d + d + -d
	r42 <- typeof(r40)
	r43 <- typeof(d)
	r44 <- typeof(ds)
	r45 <- d * d / seq_len(128) - d
	r46 <- d + d + seq_len(r40) + d + d
	r47 <- d + d + seq_len(r40) + d + d
	r48 <- d + d + seq_len(r40) + d + d
	r49 <- d + s + s + d
	r50 <- s + s + s + s + d
	
	r51 <- d + d
	r52 <- 0
	for (ii in 1:2048)
	  r52 <- r52 + 1
	  
	r53 <- d  
	for (ii in 1:64)
	  r53 <- d - r53 + d + 1
	  
	r54 <- d  
	for (ii in 1:128)
	  r54 <- d + i
	  
	r55 <- d + d
	r55 <- foo(r55 + r55,r55 + 2)
	r56 <- sum(r55)
	r57 <- sum(s)
	r58 <- r56 + r55
	
}
{
	trace.config(2)
	
	v0 <- d + d
	v1 <- d + ds
	v2 <- ds + d
	v3 <- ds + ds
	v4 <- i + i
	v5 <- i + is
	v6 <- is + i
	v7 <- is + is
	v8 <- d + 3
	v9 <- 3 + d
	v10 <- 3 + 3
	v11 <- i + 3L
	v12 <- 3L + i
	v13 <- 3L + 3L
	
	v14 <- d + i
	v15 <- i + d
	v16 <- d + is
	v17 <- d + 3L
	v18 <- is + d
	v19 <- 3L + d
	v20 <- i + ds
	v21 <- ds + i
	v22 <- 3 + i
	v23 <- i + 3
	 
	v24 <- l + d
	v25 <- d + l
	v26 <- i + l
	v27 <- l + i
	v28 <- ls + d
	v29 <- d + ls
	v30 <- i + ls
	v31 <- ls + i
	 
	v32 <- l + ds
	v33 <- ds + l
	v34 <- is + l
	v35 <- l + is
	
	v36 <- -d
	v37 <- -i
	v38 <- -ds
	v39 <- -is

	v40 <- d + d + d + d
	v41 <- d + d + d + -d
	v42 <- typeof(v40)
	v43 <- typeof(d)
	v44 <- typeof(ds)
	v45 <- d * d / seq_len(128) - d
	v46 <- d + d + seq_len(v40) + d + d
	v47 <- d + d + seq_len(v40) + d + d
	v48 <- d + d + seq_len(v40) + d + d
	v49 <- d + s + s + d
	v50 <- s + s + s + s + d
	
	v51 <- d + d
	v52 <- 0
	for (ii in 1:2048)
	  v52 <- v52 + 1
	
	v53 <- d  
	for (ii in 1:64)
	  v53 <- d - v53 + d + 1
	  
	
	v54 <- d  
	for (ii in 1:128)
	  v54 <- d + i
	  
    v55 <- d + d
	v55 <- foo(v55 + v55,v55 + 2)
	v56 <- sum(v55)
	v57 <- sum(s)
	v58 <- v56 + v55
}
 

{
	trace.config(0)
	cat("Test trace ops\n")
	PassIfEq(v0 ,  r0 )
	PassIfEq(v1  ,  r1 )
	PassIfEq(v2  ,  r2 )
	PassIfEq(v3  ,  r3 )
	PassIfEq(v4  ,  r4 )
	PassIfEq(v5  ,  r5 )
	PassIfEq(v6  ,  r6 )
	PassIfEq(v7  ,  r7 )
	PassIfEq(v8  ,  r8 )
	PassIfEq(v9  ,  r9 )
	PassIfEq(v10 ,  r10)
	PassIfEq(v11 ,  r11)
	PassIfEq(v12 ,  r12)
	PassIfEq(v13 ,  r13)
	PassIfEq(v14 ,  r14)
	PassIfEq(v15 ,  r15)
	PassIfEq(v16 ,  r16)
	PassIfEq(v17 ,  r17)
	PassIfEq(v18 ,  r18)
	PassIfEq(v19 ,  r19)
	PassIfEq(v20 ,  r20)
	PassIfEq(v21 ,  r21)
	PassIfEq(v22 ,  r22)
	PassIfEq(v23 ,  r23)
	PassIfEq(v24 ,  r24)
	PassIfEq(v25 ,  r25)
	PassIfEq(v26 ,  r26)
	PassIfEq(v27 ,  r27)
	PassIfEq(v28 ,  r28)
	PassIfEq(v29 ,  r29)
	PassIfEq(v30 ,  r30)
	PassIfEq(v31 ,  r31)
	PassIfEq(v32 ,  r32)
	PassIfEq(v33 ,  r33)
	PassIfEq(v34 ,  r34)
	PassIfEq(v35 ,  r35)
	PassIfEq(v36 ,  r36)
	PassIfEq(v37 ,  r37)
	PassIfEq(v38 ,  r38)
	PassIfEq(v39 ,  r39)
	PassIfEq(v40 ,  r40)
	PassIfEq(v41 ,  r41)
	PassIfEq(v42 ,  r42)
	PassIfEq(v43 ,  r43)
	PassIfEq(v44 ,  r44)
	PassIfEq(v45 ,  r45)
	PassIfEq(v46 ,  r46)
	PassIfEq(v47 ,  r47)
	PassIfEq(v48 ,  r48)
	PassIfEq(v49 ,  r49)
	PassIfEq(v50 ,  r50)
	PassIfEq(v51 ,  r51)
	PassIfEq(v52 ,  r52)
	PassIfEq(v53 ,  r53)
	PassIfEq(v54 ,  r54)
	PassIfEq(v55 ,  r55)
	PassIfEq(v56 ,  r56)
	PassIfEq(v57 ,  r57)
	PassIfEq(v58 ,  r58)
}

if(fail == 0)
	cat("SUCCESS! All sanity checks passed\n")
else
	cat("FAILURE! Some sanity checks failed\n")

#end the coverage in a trace to check done_op
d + d
