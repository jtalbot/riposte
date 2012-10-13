
function rw(n) 
    a = 0
    for i = 1, n do
        if math.random() < 0.5 then
            a = a+1
        else
            a = a-1
        end
    end
end

local x = os.clock()
rw(tonumber(arg[1]))
print(string.format("%f", os.clock()-x))
