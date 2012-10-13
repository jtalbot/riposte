function initial(val)
	return(math.sqrt(val))
end

function secondary(val,y)
	return(val*y)
end

function rungeKatta(t, h, N)
	local y = initial(t)
	while t < N do
		local k1 = h*secondary(t, initial(t))
		local k2 = h*secondary(t + 0.5*h, initial(t) + 0.5*k1)
		local k3 = h*secondary(t + 0.5*h, initial(t) + 0.5*k2)
		local k4 = h*secondary(t + 0.5*h, initial(t) + k3)
	    y = y + (1.0/6.0)*(k1+k2+k3+k4)
		t = t + h
	end
	return(y)
end

local x = os.clock()
rungeKatta(2,1,tonumber(arg[1]))
print(string.format("%f", os.clock() - x))
