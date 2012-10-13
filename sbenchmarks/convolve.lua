
local N = math.floor(math.sqrt(arg[1]))

function convolve(a, b)
    ab = {}
    for i = 1, 2*N do
        ab[i] = 0
    end

    for i = 1, N do
        for j = 1, N do
            ab[j+i] = ab[j+i] + a[i]*b[j]
        end
    end
end

local a = {}
local b = {}

for i = 1, N do
    a[i] = i
end

local x = os.clock()
convolve(a,a)
print(string.format("%f", os.clock() - x))
