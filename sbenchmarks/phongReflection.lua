
local ia = {233, 32, 54}
local id = {32, 34, 64}
local is = {23, 46, 78}

local normal = {1,1,1}

function addvv(x,y)
    return {x[1]+y[1],x[2]+y[2],x[3]+y[3]}
end

function subvv(x,y)
    return {x[1]-y[1],x[2]-y[2],x[3]-y[3]}
end

function dotvv(x,y)
    return x[1]*y[1]+x[2]*y[2]+x[3]*y[3]
end

function divvs(x,y)
    return {x[1]/y,x[2]/y,x[3]/y}
end

function negv(x)
    return {-x[1],-x[2],-x[3]}
end

function mulsv(x,y)
    return {x*y[1],x*y[2],x*y[3]}
end

function mulvs(x,y)
    return {x[1]*y,x[2]*y,x[3]*y}
end

function phongShading(ks, kd, ka, alpha, lightC, pointC, V)
    local lightM = subvv(lightC, pointC)
    local mag = dotvv(lightM, lightM)

    lightM = divvs(lightM, mag)
    rayM = negv(lightM)

    local iP = 
        addvv( mulsv(ka, ia),
               addvv(mulsv((kd*dotvv(lightM, normal)), id),
                     mulsv((ks*math.pow(dotvv(rayM, V), alpha)), is) ) )

    return iP
end

local lightC = {2.4, 5.6, 3.2}
local pointC = {23.5, 323.5, 434.3}
local V = {23.4, 553.3, 433.2}

function run(n)
    local total = {0,0,0}
    for i = 1,n do
        pointC[1] = pointC[1] + 1
        total = addvv(total, phongShading(2.4, 2.6, 7.3, 4.0, lightC, pointC, V))
    end
end

local x = os.clock()
run(tonumber(arg[1]))
print(string.format("%f", os.clock() - x))
