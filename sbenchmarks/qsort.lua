
function qsort(a)
   
    function qsort_kernel(lo, hi)
        local i = lo
        local j = hi
        while i < hi do
            local pivot = a[math.floor((lo+hi)/2)]
            while i <= j do
                while a[i] < pivot do i = i+1 end
                while a[j] > pivot do j = j-1 end
                if i <= j then 
                    local t = a[i]
                    a[i] = a[j]
                    a[j] = t
                    i = i+1
                    j = j-1
                end
            end

            if lo < j then qsort_kernel(lo, j) end
            lo = i
            j = hi
        end
    end 

    qsort_kernel(1, #a)
    return(a)
end

function sortperf(n) 

    a = {}
    for i = 1,n do
        a[i] = math.random()
    end

    local x = os.clock()
    qsort(a)
    print(string.format("%f", os.clock()-x))
end

sortperf(arg[1])
