
a <- 1
(a <- 1)

a <- `(`
b <- 1
a(b <- 1)

{ 1; 2 }

a <- `{`
a(1)
a(quote(1), quote(2))
