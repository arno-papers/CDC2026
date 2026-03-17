# ============================================================================
# Monod-specific plot styles and convex hull utilities.
#
# Included by Monod scripts that need DESIGN_STYLES, DESIGN_ORDER,
# convex_hull_2d(), or confidence_hull().
# ============================================================================

# ============================================================================
#  Canonical design styles
# ============================================================================

const DESIGN_STYLES = Dict(
    "adaptive"     => (label = "Adaptive policy",       color = :gray20),
    "static_std"   => (label = "Static (BIM)",           color = :dodgerblue),
    "static_spce"  => (label = "Static (sPCE-opt)",      color = :forestgreen),
)

const DESIGN_ORDER = ["adaptive", "static_std", "static_spce"]

# ============================================================================
#  2D convex hull (Graham scan) + 95% confidence region
# ============================================================================

function convex_hull_2d(px::Vector{T}, py::Vector{T}) where T
    n = length(px)
    n <= 2 && return (copy(px), copy(py))
    idx = sortperm(collect(zip(px, py)))
    sx, sy = px[idx], py[idx]
    cross(ox, oy, ax, ay, bx, by) = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
    lower_x, lower_y = T[], T[]
    for i in 1:n
        while length(lower_x) >= 2 && cross(lower_x[end-1], lower_y[end-1],
                lower_x[end], lower_y[end], sx[i], sy[i]) <= 0
            pop!(lower_x); pop!(lower_y)
        end
        push!(lower_x, sx[i]); push!(lower_y, sy[i])
    end
    upper_x, upper_y = T[], T[]
    for i in n:-1:1
        while length(upper_x) >= 2 && cross(upper_x[end-1], upper_y[end-1],
                upper_x[end], upper_y[end], sx[i], sy[i]) <= 0
            pop!(upper_x); pop!(upper_y)
        end
        push!(upper_x, sx[i]); push!(upper_y, sy[i])
    end
    hx = vcat(lower_x[1:end-1], upper_x[1:end-1])
    hy = vcat(lower_y[1:end-1], upper_y[1:end-1])
    push!(hx, hx[1]); push!(hy, hy[1])
    return (hx, hy)
end

function confidence_hull(px::AbstractVector, py::AbstractVector; frac::Float64=0.90)
    cx, cy = median(px), median(py)
    mad_x = max(median(abs.(px .- cx)), 1e-12)
    mad_y = max(median(abs.(py .- cy)), 1e-12)
    dists = [sqrt(((px[i] - cx)/mad_x)^2 + ((py[i] - cy)/mad_y)^2) for i in eachindex(px)]
    keep = sortperm(dists)[1:round(Int, frac * length(px))]
    return convex_hull_2d(Float64.(px[keep]), Float64.(py[keep]))
end
