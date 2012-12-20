
{
    a <- 0
    (repeat 
    {
        if(a == 10) break
        else a <- a+1
    })
}

{
    a <- 0
    repeat 
    {
        if(a == 10) break
        else a <- a+1
    }
    a
}

{
    a <- 0
    b <- 0
    repeat 
    {
        a <- a+1
        if(a < 10)
            next
        b <- b+1
        if(b == 10)
            break
    }
    a+b
}
