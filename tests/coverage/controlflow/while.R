
{
    a <- 0
    (while(a < 10) a <- a+1)
}

{
    a <- 0
    while(a < 10) a <- a+1
    a
}

{
    a <- 0
    while(a < 20)
    {
        a <- a+1
        if(a == 15)
            break
    }
    a
}

{
    a <- 0
    b <- 0
    while(a < 20)
    {
        a <- a+1
        if(a < 10)
            next
        b <- b+1
    }
    a+b
}
