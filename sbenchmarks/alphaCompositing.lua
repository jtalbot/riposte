
function addvv(x,y)
    return { x[1]+y[1], x[2]+y[2], x[3]+y[3] }
end

function mulvs(x,y)
    return { x[1]*y, x[2]*y, x[3]*y }
end

function alphaCompositing(color1, color2)

    return addvv( color1,
                  mulvs( color2, 1-color1[4]) )

end

function run(n)
    local color1 = {0.5,0.05,0.05,0.1}
    local color2 = {0.1,0.1,0.1,0.5}

    local a = 0
    while(a < n) do
        a = a+1
        color2 = alphaCompositing(color1, color2)
    end
    return color2
end

local x = os.clock()
run(tonumber(arg[1]))
print(string.format("%f", os.clock() - x))
