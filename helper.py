import numpy as np
import cmath
from scipy.special import gamma

STEP_SIZE = 0.025  # step size of the rectangles


# calculate roots of z^5 - 2 for visual inspection
def get_theoretical_roots():
    roots = []
    for k in range(5):
        # z^5 = 2, so z = 2^(1/5) * e^(2πik/5)
        magnitude = 2**(1/5)
        angle = 2 * np.pi * k / 5
        root = magnitude * (np.cos(angle) + 1j * np.sin(angle))
        roots.append(root)
    return roots


def winding_number(path):
    if len(path) < 3:
        return 0

    # Ensure path is closed
    closed_path = list(path)
    if closed_path[0] != closed_path[-1]:
        closed_path.append(closed_path[0])

    total_angle = 0.0

    for i in range(len(closed_path) - 1):
        z1 = closed_path[i]
        z2 = closed_path[i + 1]

        # avoid division by zero
        if abs(z1) < 1e-15 or abs(z2) < 1e-15:
            continue

        # calc angle change
        angle1 = cmath.phase(z1)
        angle2 = cmath.phase(z2)

        # avoid taking the wring angle mod 2pi
        angle_diff = angle2 - angle1
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi

        total_angle += angle_diff

    return round(total_angle / (2 * np.pi))


# create path around the rectangle
def create_rectangle_path(top_left, bottom_right):
    top_right = complex(bottom_right.real, top_left.imag)
    bottom_left = complex(top_left.real, bottom_right.imag)

    sides = [
        (top_left, top_right),
        (top_right, bottom_right),
        (bottom_right, bottom_left),
        (bottom_left, top_left)
    ]

    all_points = []
    for start, end in sides:
        distance = abs(end - start)
        num_points = max(1, int(np.ceil(distance / STEP_SIZE)))

        # interpolate along this side, to get a path with distance STEP_SIZE
        for i in range(num_points):
            t = i / num_points if num_points > 1 else 0
            point = start + t * (end - start)
            all_points.append(point)

    return all_points


def split_rectangle_at_longest_size(top_left, bottom_right):
    width = abs(bottom_right.real - top_left.real)
    height = abs(top_left.imag - bottom_right.imag)

    # Split rectangle over longest side
    if width >= height:
        # Split vertically
        mid = (top_left.real + bottom_right.real) / 2
        return [(top_left, complex(mid, bottom_right.imag)),
                (complex(mid, top_left.imag), bottom_right)]
    else:
        # Split horizontally
        mid = (top_left.imag + bottom_right.imag) / 2
        return [(top_left, complex(bottom_right.real, mid)),
                (complex(top_left.real, mid), bottom_right)]

# (-1.2544984474103784-1.2949295800668543j)
# (1.32021581008161+1.25j)
# (-0.6108198830373813-0.022464790033427162j)

def is_point_inside_rectangle(point, top_left, bottom_right):
    x = point.real
    y = point.imag

    left = min(top_left.real, bottom_right.real)
    right = max(top_left.real, bottom_right.real)
    bottom = min(top_left.imag, bottom_right.imag)
    top = max(top_left.imag, bottom_right.imag)

    return left <= x <= right and bottom <= y <= top

if __name__ == '__main__':
    tl = (-1.2544984474103784-1.2949295800668543j)
    br = (1.32021581008161+1.25j)
    point = (-0.6108198830373813-0.022464790033427162j)

    print(is_point_inside_rectangle(point, tl, br))

