function fib(N)
    local a = 0
    local b = 1
    local i = 0
    while(i < N) do
        i = i+1
        local t = b
        b = b+a
        a = t
    end
    return(b) 
end

local x = os.clock()
local n = tonumber(arg[1])
fib(n)
print(string.format("%f", os.clock() - x))
