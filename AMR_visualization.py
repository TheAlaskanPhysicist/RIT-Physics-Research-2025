import pygame as pg
import numpy as np


# Stores information about AMR blocks
class AMRBlock:
    def __init__(self, x0: int, x1: int, y0: int, y1: int, time_divisions: int):
        self.box = (x0, x1, y0, y1)
        self.time_divisions = time_divisions
    def __repr__(self):
        return f"AMRBlock({self.box}, time_divisions={self.time_divisions})"


# The class that breaks a field into AMR blocks
class AMR:
    def __init__(self, bifurcation_threshold: float = 0.5, min_block_size: int = 4):
        self.threshold = bifurcation_threshold
        self.min_size = min_block_size

    def compute_regions(self, gradient: np.ndarray, greedy=True) -> list[AMRBlock]:
        if greedy: return self._recurse_greedy(gradient, 0, gradient.shape[1], 0, gradient.shape[0])
        else:      raise NotImplementedError("Default AMR method not implemented")

    def _recurse_greedy(
            self,
            gradient: np.ndarray,
            x0: int, x1: int,
            y0: int, y1: int
    ) -> list[AMRBlock]:
        cells = []
        width = x1 - x0
        height = y1 - y0

        if width <= self.min_size and height <= self.min_size:
            return [AMRBlock(x0, x1, y0, y1, time_divisions=1)]

        subgrad = gradient[y0:y1, x0:x1]
        row_max = np.max(subgrad, axis=1) if height > 1 else np.array([0])
        col_max = np.max(subgrad, axis=0) if width > 1 else np.array([0])

        split_x = np.any(col_max > self.threshold)
        split_y = np.any(row_max > self.threshold)

        if split_x and (not split_y or width >= height):
            mx = (x0 + x1) // 2
            if mx == x0 or mx == x1:
                return [AMRBlock(x0, x1, y0, y1, time_divisions=1)]
            cells += self._recurse_greedy(gradient, x0, mx, y0, y1)
            cells += self._recurse_greedy(gradient, mx, x1, y0, y1)
        elif split_y:
            my = (y0 + y1) // 2
            if my == y0 or my == y1:
                return [AMRBlock(x0, x1, y0, y1, time_divisions=1)]
            cells += self._recurse_greedy(gradient, x0, x1, y0, my)
            cells += self._recurse_greedy(gradient, x0, x1, my, y1)
        else:
            cells.append(AMRBlock(x0, x1, y0, y1, time_divisions=1))

        return cells


# The class that handles rendering
class Renderer:
    def __init__(self, screen_resolution: tuple[int, int], grid_resolution: tuple[int, int]):
        self.screen = pg.display.set_mode(screen_resolution)
        self.WIDTH, self.HEIGHT = screen_resolution
        self.x_pixels = [int(i * (self.WIDTH / grid_resolution[0]) + 0.5) for i in range(grid_resolution[0] + 1)]
        self.y_pixels = [int(j * (self.HEIGHT / grid_resolution[1]) + 0.5) for j in range(grid_resolution[1] + 1)]

    def render_amr_blocks(self, amr_cells: list[AMRBlock], color=(255, 255, 255)):
        for block in amr_cells:
            x0, x1, y0, y1 = block.box
            px0 = self.x_pixels[x0]
            px1 = self.x_pixels[x1]
            py0 = self.y_pixels[y0]
            py1 = self.y_pixels[y1]
            pg.draw.line(self.screen, color, (px0, py0), (px1, py0), width=1)
            pg.draw.line(self.screen, color, (px0, py1), (px1, py1), width=1)
            pg.draw.line(self.screen, color, (px0, py0), (px0, py1), width=1)
            pg.draw.line(self.screen, color, (px1, py0), (px1, py1), width=1)

    def render_scalar_field(self, field: np.ndarray):
        Ny, Nx = field.shape
        for y in range(Ny):
            for x in range(Nx):
                p = field[y, x]
                p_norm = np.clip((p - 1.0) / 20.0, 0.0, 1.0)
                color = (int(p_norm * 255), 0, int((1 - p_norm) * 255))
                x0 = self.x_pixels[x]
                x1 = self.x_pixels[x + 1]
                y0 = self.y_pixels[y]
                y1 = self.y_pixels[y + 1]
                pg.draw.rect(self.screen, color, (x0, y0, x1 - x0, y1 - y0), width=0)


def main():

    # Pygame initialization
    pg.init()
    clock = pg.time.Clock()

    # Screen and simulation resolution
    screen_resolution = (800, 600)
    grid_resolution = (64*10, 64*10)

    # Create the renderer and simulation
    renderer = Renderer(screen_resolution, grid_resolution)


    # Create N random gaussians to test the AMR
    N = 10
    field = np.zeros(grid_resolution)
    for _ in range(N):
        # Random center coordinates
        cx = np.random.uniform(0, grid_resolution[0])
        cy = np.random.uniform(0, grid_resolution[1])

        # Random amplitude and standard deviation
        amplitude = np.random.uniform(50, 150)
        sigma = np.random.uniform(25, 100)

        # Apply Gaussian to entire grid
        for i in range(grid_resolution[0]):
            for j in range(grid_resolution[1]):
                dx = i - cx
                dy = j - cy
                field[i, j] += amplitude * np.exp(-(dx ** 2 + dy ** 2) / (2 * sigma ** 2))

    # Calculate the gradient
    field_gradient = np.zeros_like(field)
    for i in range(grid_resolution[0]):
        for j in range(grid_resolution[1]):
            dx = field[i, j] - field[i - 1, j] if i > 0 else 0
            dy = field[i, j] - field[i, j - 1] if j > 0 else 0
            field_gradient[i, j] = np.sqrt(dx ** 2 + dy ** 2)




    # Calculate the blocks
    amr = AMR(bifurcation_threshold=0.5, min_block_size=8)
    blocks = amr.compute_regions(field_gradient)


    # Run the render
    running = True
    while running:
        # Check if closing
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        # Render frame
        renderer.screen.fill((0, 0, 0))
        renderer.render_scalar_field(field)
        renderer.render_amr_blocks(blocks)
        pg.display.flip()
        # clock.tick(30)
        # pg.display.flip()

    # Quit Pygame
    pg.quit()

if __name__ == '__main__':
    main()