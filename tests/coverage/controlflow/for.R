
(for(i in 1:10) 1)
(for(i in 1:10) i)

{
    for(i in 1:10) i
    i
}

{
    a <- 0
    for(i in 1:10) a <- a+1
    a
}

{
    a <- 0
    b <- 1:10
    for(i in b) a <- a+i
    a
}

{
    a <- 0
    b <- 1:20
    for(i in b)
    {
        if(i > 10)
            break
        a <- a+i
    }
    a
}

{
    a <- 0
    b <- 1:10
    for(i in b)
    {
        if(i > 15)
            next
        a <- a+i
    }
    a
}
