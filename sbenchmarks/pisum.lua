
function pisum(n)
    local t=0
    for j = 1,n do
        t=0
        for k = 1,10000 do
            t = t + 1.0/(k*k)
        end
    end
    return(t)
end

local x = os.clock()
pisum(arg[1])
print(string.format("%f", os.clock()-x))
