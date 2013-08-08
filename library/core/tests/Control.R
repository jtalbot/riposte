
if(TRUE) {
    1
} else {
    2
}

a <- `if`
a(FALSE, 10, 20)

for(i in seq_len(5)) {
    print(i)
}

a <- `for`
a(i, seq_len(5), print(i))

i <- 0L
while(i < 5L) {
    print(i)
    i <- i+1L
}

a <- `while`
a(i < 5L, { print(i); i <- i+1 } )

i <- 0L
repeat {
    i <- i+1L
    if(i == 1L)
        next
    if(i > 3L)
        break
    print(i)
}

# TODO: have to deal with next & break somehow
#a <- `repeat`
#a({ i <- i+1L; if(i == 1L) next; if(i > 3L) break; print(i) })

