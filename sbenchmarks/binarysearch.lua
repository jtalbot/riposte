
function binarysearch(v, key)

    local a = 1
    local b = #v

    while a < b do
        local i = math.floor((a+b) / 2)
        if v[i] < key then
            a = i+1
        else
            b = i
        end
    end

    return(a)
end

function run(m, n)
    a = {}
    for i = 1,n do
        a[i] = i
    end
    
    for i = 1,m do
        binarysearch(a, i*(n/m))
    end
end

local x = os.clock()
run(arg[1], 10000000)
print(string.format("%f", os.clock()-x))
