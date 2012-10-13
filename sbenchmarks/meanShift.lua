

N = 8
M = 10000000/N

function vspow(v,s)
    t = {}
    for i = 0,(N-1) do
        t[i] = math.pow(v[i],s)
    end
    return(t)
end

function svdiv(s,v)
    local t = {}
    for i = 0,(N-1) do
        t[i] = s/v[i]
    end
    return(t)
end

function vvmul(v,w)
    local t = {}
    for i = 0,(N-1) do
        t[i] = v[i]*w[i]
    end
    return(t)
end

function vsmul(v,s)
    local t = {}
    for i = 0,(N-1) do
        t[i] = v[i]*s
    end
    return(t)
end

function sum(v)
    local t = 0
    for i = 0,(N-1) do
        t = t + v[i]
    end
    return(t)
end

function svsub(s,v)
    local t = {}
    for i = 0,(N-1) do
        t[i] = s-v[i]
    end
    return(t)
end

function vvdiv(v,w)
    local t = {}
    for i = 0,(N-1) do
        t[i] = v[i]/w[i]
    end
    return(t)
end

function meanshift_vector(a,x,h,d,g)
    local m = svdiv(1,vspow(h,d+2))
    local n = vvdiv( svsub(a,x), h )
          n = sum(vvmul( n, n ))
          n = n*n
    local top = sum( vsmul(vsmul(vvmul(m,x), g), n) )
    
    local m2 = svdiv(1,vspow(h,d+2))
    local n2 = vvdiv( svsub(a,x), h )
          n2 = sum(vvmul( n2, n2 ))
          n2 = n2*n2
    local bottom = sum( vsmul(vsmul(m2, g), n2) )
    
    return (top/bottom) - a
end
--function meanshift(a,x,h,d, g)
--    top <- sum((1/(h^(d+2)))*x*g*(sum((a-x)/(h) *(a-x)/(h))^2))
--    bottom <- sum((1/(h^(d+2)))*g*(sum((a-x)/(h) *(a-x)/(h))^2))
--    return(top/bottom - a)
--end

function meanshift_fused(a,x,h,d, g)
    local n = 0
    for i = 0,(N-1) do
        local t = (a-x[i])/h[i]
        n=n+t*t
    end
    n = n*n

    local t = 0
    local b = 0
    for i = 0,(N-1) do
        local q=(1/math.pow(h[i],d+2))
        t=t+q*x[i]*g*n
        b=b+q*g*n
    end
    return(t/b - a)
end

function meanshift_scalar(a,x,h,d, g)
    local n = 0
    local t = (a-x[0])/h[0]
    n=n+t*t
    n = n*n

    local t = 0
    local b = 0
    --for i = 0,(N-1) do
        local q=(1/math.pow(h[0],d+2))
        t=t+q*x[0]*g*n
        b=b+q*g*n
    --end
    return(t/b - a)
end

function meanshift_metafused(a,x,h,d,g)
    if N == 1 then 
        return(meanshift_scalar(a,x,h,d,g))
    else
        return(meanshift_fused(a,x,h,d,g))
    end
end

function seq(s, n)
    t = {}
    for i = 0, (n-1) do
        t[i] = s+i
    end
    return(t)
end

for i = 0, 16 do

N = math.pow(2,i)
M = 10000000/N

local x = seq(1001,N)
local h = seq(1,N)

function run()
    b = 0
    for i = 1,M do
        b = b + meanshift_metafused(i, x, h, b, 6)
    end
    return(b)
end

local x = os.clock()
local s = 0
run()
print(string.format("%d: elapsed time: %.4f\n", N, os.clock() - x))

end
