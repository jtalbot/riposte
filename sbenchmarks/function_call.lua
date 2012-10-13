function add(x,y)
    return(x+y)
end

function fc(N)
    local j = 1.1
    local i = 0 
    while i < N do
        i = i+1
        j = add(j,1)
    end
    return(j)
end

local x = os.clock()
local s = 0
fc(100000000)
print(string.format("elapsed time: %.4f\n", os.clock() - x))
