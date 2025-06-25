from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import colorsys

from helper import get_theoretical_roots, winding_number, create_rectangle_path, split_rectangle_at_longest_size

SCALE_MAGNITUDE_FOR_LIGHTNESS = 0.075 # avoid everything being white

# Plot size constants
LEFT_PLOT_XLIM = (-1.3, 1.3)
LEFT_PLOT_YLIM = (-1.3, 1.3)
RIGHT_PLOT_XLIM = (-20, 20)
RIGHT_PLOT_YLIM = (-20, 20)
FIGURE_SIZE = (8, 4)

ANIMATION_SPEED = 0

# Algorithm constants
MIN_SIZE = 0.5

MESSAGE_PATH, MESSAGE_RECTANGLE, MESSAGE_FILLED_RECTANGLE, MESSAGE_END = 0, 1, 2, 3

# TODO: add messages to show rectangles
# TODO: add real search

# check if zero is inside the projected rectangle
def is_zero_inside(rectangle):
    projected_rectangle = [f(point) for point in rectangle]
    
    # Check winding number around origin using the full transformed perimeter
    return abs(winding_number(projected_rectangle)) > 0

# the function to find the zeros for
def f(z):
    return z**5 - 2


def position_to_color(z):
    assert isinstance(z, complex), "Input z must be a complex number"
    magnitude = np.abs(z)
    angle = -np.angle(z, True) # use degrees for easier calculations
    lightness = np.tanh(magnitude * SCALE_MAGNITUDE_FOR_LIGHTNESS)

    # convert angle from -180, 180 to 0-1 range for hue
    hue = ((angle + 360) % 360) / 360  

    return colorsys.hls_to_rgb(hue, lightness, 0.8)

def create_search(top_left, bottom_right):
    search_queue = deque()
    search_queue.append((top_left, bottom_right))

    # start search
    while len(search_queue) > 0:
        top_left, bottom_right = search_queue.popleft()
        rectangle_points = create_rectangle_path(top_left, bottom_right)
        
        yield from [(MESSAGE_PATH, point, None) for point in rectangle_points]

        # Check if zero is inside this rectangle
        if is_zero_inside(rectangle_points):
            estimated_zero = (top_left + bottom_right) / 2
            print(f"Zero found! Estimated location: {estimated_zero:.6f}")
            print(f"  Rectangle: {top_left:.3f} to {bottom_right:.3f}")
            print(f"  Function value at estimate: {f(estimated_zero):.6f}")

            yield MESSAGE_RECTANGLE, top_left, bottom_right
            
            # Only split if rectangle is large enough
            width = abs(bottom_right.real - top_left.real)
            height = abs(top_left.imag - bottom_right.imag)

            if width > MIN_SIZE and height > MIN_SIZE:
                rectangles = split_rectangle_at_longest_size(top_left, bottom_right)
                if rectangles:
                    for rectangle in rectangles:
                        search_queue.append(rectangle)
        else:
            yield MESSAGE_FILLED_RECTANGLE, top_left, bottom_right
    yield MESSAGE_END, None, None


def create_animated_plot():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGURE_SIZE)
    
    # Add main title to the figure
    fig.suptitle('Complex Roots of z⁵ - 2', fontsize=14, y=0.95)
    
    # Left subplot - accumulated search pattern
    ax1.set_xlim(*LEFT_PLOT_XLIM)
    ax1.set_ylim(*LEFT_PLOT_YLIM)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_title('Domain')
    
    # Right subplot - current point only
    ax2.set_xlim(*RIGHT_PLOT_XLIM)
    ax2.set_ylim(*RIGHT_PLOT_YLIM)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.set_title('Projection')
    
    # Create HSL background gradient for right subplot
    x_range = np.linspace(*RIGHT_PLOT_XLIM, 200)
    y_range = np.linspace(*RIGHT_PLOT_YLIM, 200)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Create color array for background with better scaling
    background_colors = np.zeros((len(y_range), len(x_range), 3))
    for i, y in enumerate(y_range):  # Don't reverse y here, let imshow handle orientation
        for j, x in enumerate(x_range):
            z = complex(x, y)
            # Use the same position_to_color function for consistency
            color = position_to_color(z)
            background_colors[i, j] = color
    
    # Display background gradient
    ax2.imshow(background_colors, extent=[*RIGHT_PLOT_XLIM, *RIGHT_PLOT_YLIM], origin='lower', alpha=0.8)
    
    # Plot theoretical roots as red crosses
    # theoretical_roots = get_theoretical_roots()
    # root_x = [root.real for root in theoretical_roots]
    # root_y = [root.imag for root in theoretical_roots]
    # ax1.scatter(root_x, root_y, c='red', marker='x', s=100, linewidths=3, label='Theoretical roots', zorder=10)
    
    # Create scatter plot objects for left subplot
    scatter_left = ax1.scatter([], [], s=2, alpha=0.7, label='Search points')
    current_point_left = ax1.scatter([], [], s=20, label='Current point', zorder=5)
    
    # Create scatter plot object for right subplot (current point only)
    current_point_right = ax2.scatter([], [], s=100, zorder=5)
    
    # Add legend to left subplot

    # Create circle generator and storage for accumulated points
    search_gen = create_search(complex(LEFT_PLOT_XLIM[0] * 0.95 , LEFT_PLOT_YLIM[0] * 0.95), complex(LEFT_PLOT_XLIM[1] * 0.95, LEFT_PLOT_YLIM[1] * 0.95))
    accumulated_points = []

    rectangles = []
    filled_rectangles = []
    filled_patches = []  # Store Rectangle patches for filled rectangles
    rectangle_patches = {}  # Store outline Rectangle patches for rectangles
    
    def animate(frame):
        try:
            # generate next points, until done

            next_point = None
            message_type, message_value1, message_value2 = next(search_gen)

            # TODO: add done message to remove current_point

            if message_type == MESSAGE_PATH:
                next_point = message_value1
                accumulated_points.append(next_point)
            elif message_type == MESSAGE_RECTANGLE:
                accumulated_points.clear()
                rectangles.append((message_value1, message_value2))
                # Add outline rectangle patch if not already present
                key = (message_value1, message_value2)
                if key not in rectangle_patches:
                    x = message_value1.real
                    y = message_value1.imag
                    width = message_value2.real - message_value1.real
                    height = message_value2.imag - message_value1.imag
                    rect_patch = plt.Rectangle((x, y), width, height, fill=False, edgecolor='#222', linewidth=1, zorder=0.5)
                    ax1.add_patch(rect_patch)
                    rectangle_patches[key] = rect_patch
                # TODO: implement with dictionary, also add message to remove rectangle again
            elif message_type == MESSAGE_FILLED_RECTANGLE:
                accumulated_points.clear()
                filled_rectangles.append((message_value1, message_value2))
            elif message_type == MESSAGE_END:
                accumulated_points.clear()
                print("END ANIMATION")

        except StopIteration:
            pass

        # Only add new filled rectangle patches
        for i in range(len(filled_patches), len(filled_rectangles)):
            top_left, bottom_right = filled_rectangles[i]
            x = top_left.real
            y = top_left.imag
            width = bottom_right.real - top_left.real
            height = bottom_right.imag - top_left.imag
            rect_patch = plt.Rectangle((x, y), width, height, color='black', alpha=0.3, zorder=1)
            ax1.add_patch(rect_patch)
            filled_patches.append(rect_patch)

        if len(accumulated_points) > 0:
            # TODO: no need to recalculate every time, just let the generator calculate the color
            # Extract x, y coordinates and generate colors
            x_current = [point.real for point in accumulated_points]
            y_current = [point.imag for point in accumulated_points]
            colors = [position_to_color(f(point)) for point in accumulated_points]
            
            # Update left subplot - all accumulated points
            scatter_left.set_offsets(np.column_stack((x_current, y_current)))
            scatter_left.set_color(colors)
            
            # Update current point in left subplot
            current_point_left.set_offsets([[x_current[-1], y_current[-1]]])
            current_color = position_to_color(f(accumulated_points[-1]))
            current_point_left.set_color([current_color])
            
            # Update right subplot - show projection of f(point)
            projection = f(accumulated_points[-1])
            current_point_right.set_offsets([[projection.real, projection.imag]])
            projection_color = position_to_color(projection)
            current_point_right.set_color([projection_color])
        else:
            # Hide the current point and accumulated points
            scatter_left.set_offsets(np.empty((0, 2)))
            current_point_left.set_offsets(np.empty((0, 2)))
            current_point_right.set_offsets(np.empty((0, 2)))

        # Return all artists including outline rectangles
        return [scatter_left, current_point_left, current_point_right] + filled_patches + list(rectangle_patches.values())
    
    # Create animation with explicit cache_frame_data=False for unknown length
    anim = animation.FuncAnimation(
        fig, animate, interval=ANIMATION_SPEED, blit=True, repeat=True, cache_frame_data=False
    )

    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == '__main__':
    animation_obj = create_animated_plot()