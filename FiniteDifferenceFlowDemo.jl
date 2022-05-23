using GLMakie, LinearAlgebra, Tullio, OffsetArrays

struct Params
    dx::Float64
    dy::Float64
    dt::Float64
    x::Int64
    y::Int64
    t::Int64
    ρ::Float64
    ν::Float64
end

function pressure_poisson_solve!(n,p,u,v,po)
    (; dx, dy, dt, x, y, ρ, ν) = po
    for i in 2:x-1
        for j in 2:y-1
            
            p[n,i,j] = begin (
                + 1/(2 * (dx^2+dy^2)) * (
                    + dy^2 * (p[n,i+1,j] + p[n,i-1,j])
                    + dx^2 * (p[n,i,j+1] + p[n,i,j-1])
                )
                - (ρ*dx^2*dy^2) / (2 * (dx^2 + dy^2)) * (
                    + 1/dt * (
                        + (u[n,i+1,j] - u[n,i-1,j]) / (2*dx)
                        + (v[n,i,j+1] - v[n,i,j-1]) / (2*dy)
                    )
                    - ((u[n,i+1,j] - u[n,i-1,j]) / (2*dx))^2
                    - ((v[n,i,j+1] - v[n,i,j-1]) / (2*dy))^2
                    - 2 * (
                        + ((u[n,i,j+1] - u[n,i,j-1]) / (2*dy))
                        * ((v[n,i+1,j] - v[n,i-1,j]) / (2*dx))
                    )
                )
                )
            end
        end
    end
    return nothing
end

function sim!(p,u,v,po)
    #New to Julia 1.7, we can load parameters with the below syntax
    (; dx, dy, dt, x, y, t, ρ, ν) = po
    for n in 1:t-1
        pressure_poisson_solve!(n,p,u,v,po)
        for i in 2:x-1
            for j in 2:y-1
                #This is the alternating "spigot" inital conditions
                mod1(n,100) > 50 ? v[n,50:52,50:52] .+= cos(n/(2pi)) * 1e-5 :
                    v[n,50:52,50:52] .+= sin(n/(2pi)) * 1e-5
                mod1(n,100) > 50 ? u[n,50:52,50:52] .+= sin(n/(2pi)) * 1e-5 :
                    u[n,50:52,50:52] .+= cos(n/(2pi)) * 1e-5

                u[n+1,i,j] = begin (
                    + u[n,i,j]
                    - u[n,i,j] * dt/dx * (u[n,i,j] - u[n,i-1,j])
                    - v[n,i,j] * dt/dy * (u[n,i,j] - u[n,i,j-1])
                    - dt/(ρ*2*dx) * (p[n,i+1,j] - p[n,i-1,j])
                    + ν * (
                        + dt/dx^2 * (u[n,i+1,j] - 2*u[n,i,j] + u[n,i-1,j])
                        + dt/dy^2 * (u[n,i,j+1] - 2*u[n,i,j] + u[n,i,j-1])
                    )
                    )
                end
                
                v[n+1,i,j] = begin (
                    + v[n,i,j]
                    - u[n,i,j] * dt/dx * (v[n,i,j] - v[n,i-1,j])
                    - v[n,i,j] * dt/dy * (v[n,i,j] - v[n,i,j-1])
                    - dt/(ρ*2*dy) * (p[n,i,j+1] - p[n,i,j-1])
                    + ν * (
                        + dt/dx^2 * (v[n,i+1,j] - 2*v[n,i,j] + v[n,i-1,j])
                        + dt/dy^2 * (v[n,i,j+1] - 2*v[n,i,j] + v[n,i,j-1])
                    )
                    )
                end
            end
        end
    end
    return nothing
end

begin
    #x and y define grid size. t is the number of timesteps
    x = 100
    y =100
    t = 2000

    #These define legnth of time steps. 
    #Careful, going to far from these values can yield numerical instability for certain parameter regimes.
    dt = 0.005
    dx =0.1
    dy = 0.1

    #ρ is the gas density term, ν is viscosity. 
    ρ = 1
    ν = 0.51
    
    #initializing the output arrays.
    u = zeros(t,x,y)
    v = zeros(t,x,y)
    p = zeros(t,x,y)

    #Parameter struct and main function call.
    # Having a parameter struct is necessary for proper type inference.
    #If you modify sim! with global variables check code_warntype to make surface
    #Julia knows what type you are using or you will suffa 10x slowdown. 
    po = Params(dx, dy, dt, x,y,t, ρ,ν)
    @time sim!(p,u,v,po)
end

begin
    #Here we collect our output 3D p,u,v arrays into a 1D array of 2D matrices
    pressure_array = collect(eachslice(p,dims=1))
    u_array = collect(eachslice(u,dims=1))
    v_array = collect(eachslice(v,dims=1))
    
    
    #we apply a gaussian filter to our pressure array to smooth
    #it out a little and make it more visually appealling
    filter_1d = OffsetArray([1, 2, 1], -1:1) 
    filter_2d = filter_1d*filter_1d'
    pressure_filtered = [(@tullio h[i,j] := m[i+k,j+l] * filter_2d[k,l]) |> OffsetArrays.no_offset_view for m in pressure_array]
    
    #value n describes the density of arrows in the quiver plot
    n= 2
    xs = 1:n:100 |> collect
    ys = 1:n:100 |> collect
    
    #"T" is the Makie observable we use to loop through our arrays for visualization
    T = Observable(1)

    #Here we are making sure our velocity arrows will fit in their proper locations given a "n" value.
    us = [[k[i,j] for i in xs, j in ys] for k in u_array]
    vs = [[k[i,j] for i in xs, j in ys] for k in v_array]
    
    #When "T" is updated in the record loop, these values update wherever they are referenced
    us1 = @lift(us[$T])
    vs1 = @lift(vs[$T])
    ps1 = @lift(pressure_filtered[$T])
    
    
    
    fig, ax, pl = heatmap(1:100,1:100,ps1,
        colormap=:curl, colorrange=(-20,20))
    
    #uncomment below to see the velocity field
    #arrows!(ax, xs, ys,us1,vs1)
end

#replace 1:t with arbitrary range to watch it loop
record(fig, "flow.mp4", 1:1000) do i
    T[] = mod1(i,size(pressure_array,1))
end