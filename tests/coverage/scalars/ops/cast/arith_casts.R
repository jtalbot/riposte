# tests implicit casting due to arithmetic operations

-TRUE
#DIFF: R results in FALSE, Riposte results in 0 
#+FALSE

-5L
+10L

1L+1
1+1L

TRUE+1L
1L+FALSE

FALSE+2
2+TRUE

TRUE*TRUE
TRUE*FALSE

3L^2L
2L^4
2^3L

3L/2L
3L%/%2L

FALSE/TRUE
FALSE%/%TRUE

FALSE/5L
TRUE%/%10L

5/TRUE
5%/%TRUE
