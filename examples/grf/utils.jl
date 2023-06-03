

function convert_to_animation(x, time_stride, N, clims)
    init_frames = length(x)
    x = x[1:time_stride:init_frames]
    frames = length(x)
    animation = @animate for i = 1:frames
            heatmap(
                reshape(x[i],(N,N)),
                xaxis = false, yaxis = false, xticks = false, yticks = false,
                clims = clims
            )
    end
    return animation
end
