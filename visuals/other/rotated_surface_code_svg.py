# rotated_surface_code_svg_v2.py
def generate_rotated_surface_code_svg(
    d: int,
    path: str,
    spacing: int = 100,
    margin: int = 80,
    data_radius: int = 8,
    anc_size: int = 14,
    stroke_width: int = 2,
    title: str | None = None,
    # Boundary alternation phases:
    # - Top/Bottom are X-type; Left/Right are Z-type.
    # - Phases choose which edge *segments* get ancillas so none are adjacent.
    top_phase: int = 0,     # top X on segments where i % 2 == top_phase
    bottom_phase: int = 1,  # bottom X on segments where i % 2 == bottom_phase
    left_phase: int = 1,    # left Z on segments where j % 2 == left_phase
    right_phase: int = 0,   # right Z on segments where j % 2 == right_phase
) -> None:
    """
    Generate an SVG of a rotated surface code with distance d and save it to 'path'.

    - Data qubits at integer grid (i,j) for i,j in [0..d-1].
    - Interior plaquettes at (i+0.5, j+0.5), checkerboard types:
        (i + j) % 2 == 0 -> Z (diamond), else X (square).
    - Boundary plaquettes (alternating, nonadjacent):
        * Top/Bottom: X-type at (i+0.5, -0.5) and (i+0.5, d-0.5)
        * Left/Right:  Z-type at (-0.5, j+0.5) and (d-0.5, j+0.5)
    """
    assert d >= 2, "Distance d must be >= 2"
    assert top_phase in (0,1) and bottom_phase in (0,1) and left_phase in (0,1) and right_phase in (0,1)

    # Canvas includes half-spacing on each side for boundary checks
    width = margin*2 + spacing*(d-1) + spacing
    height = margin*2 + spacing*(d-1) + spacing

    def data_pos(i, j):
        return margin + i * spacing, margin + j * spacing

    parts = []
    parts.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, Arial, sans-serif">\n'
    )
    # Add a solid white background so no transparency
    parts.append(f'<rect width="100%" height="100%" fill="white"/>\n')
    shift_y = 40  # move everything down 40px
    parts.append(f'<g transform="translate(0,{shift_y})">\n')

    # Light grid
    end_x = margin + (d-1)*spacing
    end_y = margin + (d-1)*spacing

    # Light grid (clamped to lattice)
    for j in range(d):
        y = margin + j*spacing
        parts.append(f'<line x1="{margin}" y1="{y}" x2="{end_x}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>\n')
    for i in range(d):
        x = margin + i*spacing
        parts.append(f'<line x1="{x}" y1="{margin}" x2="{x}" y2="{end_y}" stroke="#e0e0e0" stroke-width="1"/>\n')

   # if title is None:
        #title = f"Rotated Surface Code (d = {d})"
    parts.append(f'<text x="{width/2}" y="30" font-size="18" text-anchor="middle" fill="#222"></text>\n')

    # Interior plaquettes
    for j in range(d-1):
        for i in range(d-1):
            cx = margin + (i+0.5)*spacing
            cy = margin + (j+0.5)*spacing
            ptype = 'Z' if (i + j) % 2 == 0 else 'X'
            # Connect to neighbors
            for (ii, jj) in [(i,j),(i+1,j),(i,j+1),(i+1,j+1)]:
                x, y = data_pos(ii, jj)
                parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#666" stroke-width="{stroke_width}" opacity="0.6"/>\n')
            # Draw ancilla
            if ptype == 'Z':
                pts = [(cx, cy - anc_size), (cx + anc_size, cy), (cx, cy + anc_size), (cx - anc_size, cy)]
                pts_str = " ".join(f"{x},{y}" for x,y in pts)
                parts.append(f'<polygon points="{pts_str}" fill="#FD6360" stroke="#cc3333" stroke-width="{stroke_width}"/>\n')
                parts.append(f'<text x="{cx}" y="{cy+5}" font-size="14" text-anchor="middle" fill="#991111">Z</text>\n')
            else:
                x0, y0 = cx - anc_size, cy - anc_size
                parts.append(f'<rect x="{x0}" y="{y0}" width="{2*anc_size}" height="{2*anc_size}" fill="#46A5FF" stroke="#3366cc" stroke-width="{stroke_width}"/>\n')
                parts.append(f'<text x="{cx}" y="{cy+5}" font-size="14" text-anchor="middle" fill="#113399">X</text>\n')

    # Boundary plaquettes — alternating, nonadjacent
    # Top (X)
    for i in range(d-1):
        if (i % 2) != top_phase:
            continue
        cx = margin + (i+0.5)*spacing
        cy = margin - 0.5*spacing
        for (ii, jj) in [(i,0), (i+1,0)]:
            x, y = data_pos(ii, jj)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#666" stroke-width="{stroke_width}" opacity="0.6"/>\n')
        x0, y0 = cx - anc_size, cy - anc_size
        parts.append(f'<rect x="{x0}" y="{y0}" width="{2*anc_size}" height="{2*anc_size}" fill="#46A5FF" stroke="#3366cc" stroke-width="{stroke_width}"/>\n')
        parts.append(f'<text x="{cx}" y="{cy+5}" font-size="14" text-anchor="middle" fill="#113399">X</text>\n')

    # Bottom (X)
    for i in range(d-1):
        if (i % 2) != bottom_phase:
            continue
        cx = margin + (i+0.5)*spacing
        cy = margin + (d-1 + 0.5)*spacing
        for (ii, jj) in [(i,d-1), (i+1,d-1)]:
            x, y = data_pos(ii, jj)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#666" stroke-width="{stroke_width}" opacity="0.6"/>\n')
        x0, y0 = cx - anc_size, cy - anc_size
        parts.append(f'<rect x="{x0}" y="{y0}" width="{2*anc_size}" height="{2*anc_size}" fill="#46A5FF" stroke="#3366cc" stroke-width="{stroke_width}"/>\n')
        parts.append(f'<text x="{cx}" y="{cy+5}" font-size="14" text-anchor="middle" fill="#113399">X</text>\n')

    # Left (Z)
    for j in range(d-1):
        if (j % 2) != left_phase:
            continue
        cx = margin - 0.5*spacing
        cy = margin + (j+0.5)*spacing
        for (ii, jj) in [(0,j), (0,j+1)]:
            x, y = data_pos(ii, jj)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#666" stroke-width="{stroke_width}" opacity="0.6"/>\n')
        pts = [(cx, cy - anc_size), (cx + anc_size, cy), (cx, cy + anc_size), (cx - anc_size, cy)]
        pts_str = " ".join(f"{x},{y}" for x,y in pts)
        parts.append(f'<polygon points="{pts_str}" fill="#FD6360" stroke="#cc3333" stroke-width="{stroke_width}"/>\n')
        parts.append(f'<text x="{cx}" y="{cy+5}" font-size="14" text-anchor="middle" fill="#991111">Z</text>\n')

    # Right (Z)
    for j in range(d-1):
        if (j % 2) != right_phase:
            continue
        cx = margin + (d-1 + 0.5)*spacing
        cy = margin + (j+0.5)*spacing
        for (ii, jj) in [(d-1,j), (d-1,j+1)]:
            x, y = data_pos(ii, jj)
            parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x}" y2="{y}" stroke="#666" stroke-width="{stroke_width}" opacity="0.6"/>\n')
        pts = [(cx, cy - anc_size), (cx + anc_size, cy), (cx, cy + anc_size), (cx - anc_size, cy)]
        pts_str = " ".join(f"{x},{y}" for x,y in pts)
        parts.append(f'<polygon points="{pts_str}" fill="#FD6360" stroke="#cc3333" stroke-width="{stroke_width}"/>\n')
        parts.append(f'<text x="{cx}" y="{cy+5}" font-size="14" text-anchor="middle" fill="#991111">Z</text>\n')

    # Data qubits
    for j in range(d):
        for i in range(d):
            x, y = data_pos(i, j)
            parts.append(f'<circle cx="{x}" cy="{y}" r="{data_radius}" fill="black"/>\n')

    # Boundary labels (X = rough top/bottom, Z = smooth left/right)
    for j in range(d):
        for i in range(d):
            x, y = data_pos(i, j)
            parts.append(f'<circle cx="{x}" cy="{y}" r="{data_radius}" fill="black"/>\n')
    parts.append('</g>\n')
    parts.append('</svg>\n')

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))



generate_rotated_surface_code_svg(
    5, "rot_5.svg",
    top_phase=0,    # top X above d00–d10
    bottom_phase=1, # bottom X below d12–d22
    left_phase=1,   # left Z beside d02–d01
    right_phase=0   # right Z beside d20–d21
)

from cairosvg import svg2png
svg2png(url="rot_5.svg", write_to="surface_code_d5.png")
