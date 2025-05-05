import pygame as pg
import numpy as np
from typing import List, Tuple

Rect = Tuple[int, int, int, int]  # (x0, x1, y0, y1)

class AMRBlock:
    def __init__(self, x0: int, x1: int, y0: int, y1: int, time_divisions: int):
        self.box = (x0, x1, y0, y1)
        self.time_divisions = time_divisions

class AMR:
    def __init__(self, bifurcation_threshold: float = 0.5, min_block_size: int = 4):
        self.threshold = bifurcation_threshold
        self.min_size = min_block_size

    def compute_regions(self, gradient: np.ndarray, greedy=True) -> List[AMRBlock]:
        if greedy:
            return self._recurse_greedy(gradient, 0, gradient.shape[1], 0, gradient.shape[0], degree=0)
        else:
            return self._recurse_default(gradient, 0, gradient.shape[1], 0, gradient.shape[0])

    def _recurse_greedy(
            self,
            gradient: np.ndarray,
            x0: int, x1: int,
            y0: int, y1: int,
            degree: int
    ) -> List[AMRBlock]:
        cells = []
        width = x1 - x0
        height = y1 - y0

        if width <= self.min_size and height <= self.min_size:
            max_gradient = np.max(gradient[y0:y1, x0:x1])
            return [AMRBlock(x0, x1, y0, y1, time_divisions=abs(max_gradient)*pow(2,degree))]

        subgrad = gradient[y0:y1, x0:x1]
        row_max = np.max(subgrad, axis=1) if height > 1 else np.array([0])
        col_max = np.max(subgrad, axis=0) if width > 1 else np.array([0])

        split_x = np.any(col_max > self.threshold)
        split_y = np.any(row_max > self.threshold)

        if split_x and (not split_y or width >= height):
            mx = (x0 + x1) // 2
            if mx == x0 or mx == x1:
                return [AMRBlock(x0, x1, y0, y1, time_divisions=(abs(np.max(subgrad)))*pow(2,degree))]
            cells += self._recurse_greedy(gradient, x0, mx, y0, y1, degree + 1)
            cells += self._recurse_greedy(gradient, mx, x1, y0, y1, degree + 1)
        elif split_y:
            my = (y0 + y1) // 2
            if my == y0 or my == y1:
                return [AMRBlock(x0, x1, y0, y1, time_divisions=(abs(np.max(subgrad)))*pow(2,degree))]
            cells += self._recurse_greedy(gradient, x0, x1, y0, my, degree + 1)
            cells += self._recurse_greedy(gradient, x0, x1, my, y1, degree + 1)
        else:
            cells.append(AMRBlock(x0, x1, y0, y1, time_divisions=(abs(np.max(subgrad)))*pow(2,degree)))

        return cells

    def compute_uniform_regions(self, gradient: np.ndarray):
        # Break into uniform blocks
        cells = []
        width = gradient.shape[1]
        height = gradient.shape[0]

        for x0 in range(0, width, self.min_size):
            x1 = min(x0 + self.min_size, width)
            for y0 in range(0, height, self.min_size):
                y1 = min(y0 + self.min_size, height)
                cells.append(AMRBlock(x0, x1, y0, y1, time_divisions=abs(np.max(gradient[y0:y1, x0:x1]))))

        return cells





    def _recurse_default(
        self,
        gradient: np.ndarray,
        x0: int, x1: int,
        y0: int, y1: int
    ) -> List[AMRBlock]:
        raise NotImplementedError()

class Renderer:
    def __init__(self, screen_resolution: tuple[int, int], grid_resolution: tuple[int, int]):
        self.screen = pg.display.set_mode(screen_resolution)
        self.WIDTH, self.HEIGHT = screen_resolution
        self.x_pixels = [int(i * (self.WIDTH / grid_resolution[0]) + 0.5) for i in range(grid_resolution[0] + 1)]
        self.y_pixels = [int(j * (self.HEIGHT / grid_resolution[1]) + 0.5) for j in range(grid_resolution[1] + 1)]

    def render_amr_cells(self, amr_cells: List[AMRBlock], color=(255, 255, 255), width=1):
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

    def render_dts_cells(self, amr_cells: List[AMRBlock]):
        for block in amr_cells:
            x0, x1, y0, y1 = block.box
            px0 = self.x_pixels[x0]
            px1 = self.x_pixels[x1]
            py0 = self.y_pixels[y0]
            py1 = self.y_pixels[y1]
            # p_norm = np.clip(np.sqrt((block.time_divisions) / 4096), 0.0, 1.0)
            p_norm = np.clip((block.time_divisions / 0.75), 0.0, 1.0)

            color = (int(p_norm * 255), int(p_norm * 255), int(p_norm * 255))
            # Draw a filled rectangle
            pg.draw.rect(self.screen, color, (px0, py0, px1 - px0, py1 - py0), width=0)

    def render_pressure_field(self, field: np.ndarray):
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




class Simulation:
    def __init__(self, grid_resolution: tuple[int, int]):
        self.WIDTH, self.HEIGHT = grid_resolution

        # Scalar field and gradient
        self.pressure_field = np.zeros((self.WIDTH, self.HEIGHT), dtype=np.uint8)
        self.pressure_field_gradient = None

        # Conserved scalar fields
        self.rho = np.ones(grid_resolution)
        self.rho_ux = np.zeros(grid_resolution)
        self.rho_uy = np.zeros(grid_resolution)
        self.total_energy = np.ones(grid_resolution) * 2.5

        # Default field
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        for y in range(self.HEIGHT):
            for x in range(self.WIDTH):
                dist2 = (x - cx) ** 2 + (y - cy) ** 2
                self.total_energy[y, x] += 50.0 * np.exp(-dist2 ** 2 / 10000.0)
        # Make a vertical strip of pressure that's a few pixels wide
        # self.total_energy[:, self.WIDTH // 4] += 50.0
        # self.total_energy[self.HEIGHT - 10:self.HEIGHT - 5, self.WIDTH - 10:self.WIDTH - 5] += 50.0


        # AMR management
        self.amr = AMR(bifurcation_threshold=0.5, min_block_size=1)
        self.blocks = None

        # Time evolution
        self.time = 0.0
        self.diffusion_rate = 1.0

    def pressure(self, rho, rho_ux, rho_uy, E):
        GAMMA = 1.4
        kinetic_energy = 0.5 * (rho_ux**2 + rho_uy**2) / rho
        return (GAMMA - 1.0) * (kinetic_energy + E)




    def update(self, dt=0.1, greedy=True):

        # Pressure equation of state
        self.pressure_field = self.pressure(self.rho, self.rho_ux, self.rho_uy, self.total_energy)
        self.pressure_field_gradient = compute_gradient(self.pressure_field)
        self.blocks = self.amr.compute_regions(self.pressure_field_gradient, greedy)
        # self.blocks = self.amr.compute_uniform_regions(self.pressure_field_gradient)

        # Make a copy of the conserved quantities
        new_rho = self.rho.copy()
        new_rho_ux = self.rho_ux.copy()
        new_rho_uy = self.rho_uy.copy()
        new_total_energy = self.total_energy.copy()

        # Iterate over AMR blocks
        for block in self.blocks:
            x0, x1, y0, y1 = block.box

            rho = self.rho[y0:y1, x0:x1]
            rho_ux = self.rho_ux[y0:y1, x0:x1]
            rho_uy = self.rho_uy[y0:y1, x0:x1]
            E = self.total_energy[y0:y1, x0:x1]

            u = rho_ux / rho
            v = rho_uy / rho
            p = self.pressure(rho, rho_ux, rho_uy, E)

            padded_rho = np.pad(rho, 1, mode='edge')
            padded_rho_ux = np.pad(rho_ux, 1, mode='edge')
            padded_rho_uy = np.pad(rho_uy, 1, mode='edge')
            padded_E = np.pad(E, 1, mode='edge')
            padded_p = np.pad(p, 1, mode='edge')

            flux_rho = -0.5 * (padded_rho[2:, 1:-1] - padded_rho[:-2, 1:-1])
            flux_rho_ux = -0.5 * (padded_rho_ux[2:, 1:-1] - padded_rho_ux[:-2, 1:-1])
            flux_rho_uy = -0.5 * (padded_rho_uy[2:, 1:-1] - padded_rho_uy[:-2, 1:-1])
            flux_E = -0.5 * (padded_E[2:, 1:-1] - padded_E[:-2, 1:-1])

            new_rho[y0:y1, x0:x1] += dt * flux_rho
            new_rho_ux[y0:y1, x0:x1] += dt * flux_rho_ux
            new_rho_uy[y0:y1, x0:x1] += dt * flux_rho_uy
            new_total_energy[y0:y1, x0:x1] += dt * flux_E

        # Save to state
        self.rho = new_rho
        self.rho_ux = new_rho_ux
        self.rho_uy = new_rho_uy
        self.total_energy = new_total_energy
        self.time += dt






def generate_test_field(shape: Tuple[int, int]) -> np.ndarray:
    Ny, Nx = shape
    field = np.ones((Ny, Nx))
    cx, cy = Nx // 2, Ny // 2
    for y in range(Ny):
        for x in range(Nx):
            dist2 = (x - cx)**2 + (y - cy)**2
            field[y, x] += 10.0 * np.exp(-dist2 / 200.0)
    field[:, Nx // 4] += 5.0
    field[Ny - 10:Ny - 5, Nx - 10:Nx - 5] += 15.0
    return field

def compute_gradient(field: np.ndarray) -> np.ndarray:
    dpdx = np.zeros_like(field)
    dpdy = np.zeros_like(field)
    dpdx[:, 1:-1] = (field[:, 2:] - field[:, :-2]) / 2
    dpdy[1:-1, :] = (field[2:, :] - field[:-2, :]) / 2
    return np.sqrt(dpdx**2 + dpdy**2)





def main():

    # Pygame initialization
    pg.init()
    clock = pg.time.Clock()

    # Screen and simulation resolution
    screen_resolution = (800, 600)
    grid_resolution = (64, 64)

    # Create the renderer and simulation
    renderer = Renderer(screen_resolution, grid_resolution)
    simulation = Simulation(grid_resolution)
    simulation.update(dt=0.1, greedy=True)

    # Run the simulation
    running = True
    while running:
        # Check if closing
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False

        # Simulate next step
        # simulation.update(dt=0.1, greedy=True)

        # Render frame
        renderer.screen.fill((0, 0, 0))
        renderer.render_pressure_field(simulation.pressure_field)
        # renderer.render_dts_cells(simulation.blocks)
        # renderer.render_amr_cells(simulation.blocks)
        pg.display.flip()
        clock.tick(30)

    # Quit Pygame
    pg.quit()

if __name__ == '__main__':
    main()