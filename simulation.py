import pygame
import numpy as np
import random
import sys
import json

# Initialize Pygame
pygame.init()

# --- Configurations & Hyperparameters ---
GRID_SIZE = 20          
CELL_SIZE = 35          
UI_HEIGHT = 120         
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = (GRID_SIZE * CELL_SIZE) + UI_HEIGHT

FPS_NORMAL = 10         
FPS_FAST = 500          

# Color Palette (Dark Theme)
COLOR_BG = (18, 18, 24)
COLOR_WALL = (44, 44, 56)
COLOR_PATH = (28, 28, 36)
COLOR_AVATAR = (41, 128, 185)      
COLOR_FUSED = (155, 89, 182)       
COLOR_PREDATOR = (231, 76, 60)     
COLOR_TEXT = (236, 240, 241)
COLOR_LASER = (231, 76, 60)

# --- Neural Network (The Shared Mind) ---
class SharedMind:
    def __init__(self, input_size=12, hidden_size=16, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, x):
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out

    def mutate(self, rate=0.08):
        mutated = SharedMind(self.input_size, self.hidden_size, self.output_size)
        mutated.W1 = self.W1 + np.random.randn(*self.W1.shape) * rate
        mutated.b1 = self.b1 + np.random.randn(*self.b1.shape) * rate
        mutated.W2 = self.W2 + np.random.randn(*self.W2.shape) * rate
        mutated.b2 = self.b2 + np.random.randn(*self.b2.shape) * rate
        return mutated

    def get_attention_profile(self):
        """
        Calculates the average weight magnitude connected to Walls, Avatars, and Predator.
        Shows what the network is prioritizing mathematically.
        """
        wall_indices = [0, 3, 6, 9]
        avatar_indices = [1, 4, 7, 10]
        predator_indices = [2, 5, 8, 11]
        
        wall_att = float(np.mean(np.abs(self.W1[wall_indices, :])))
        avatar_att = float(np.mean(np.abs(self.W1[avatar_indices, :])))
        predator_att = float(np.mean(np.abs(self.W1[predator_indices, :])))
        
        return {
            "wall_attention": round(wall_att, 4),
            "avatar_attention": round(avatar_att, 4),
            "predator_attention": round(predator_att, 4)
        }

# --- Avatar Class ---
class Avatar:
    def __init__(self, pos):
        self.pos = pos      
        self.size = 1       

# --- Maze Generation ---
def create_maze():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[5, 2:8] = 1
    grid[5, 12:18] = 1
    grid[14, 2:8] = 1
    grid[14, 12:18] = 1
    grid[2:8, 10] = 1
    grid[12:18, 10] = 1
    grid[9, 4:16] = 1
    grid[9, 10] = 0
    grid[5, 5] = 0
    grid[14, 14] = 0
    return grid

# --- Main Simulation Engine ---
class Simulation:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Neural Maze Simulation (Data Logging)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)
        self.font_bold = pygame.font.SysFont("Arial", 18, bold=True)
        
        self.grid = create_maze()
        
        # Evolutionary variables
        self.best_mind = SharedMind()
        self.best_fitness = -999999
        self.generation = 1
        self.candidate_mind = self.best_mind  
        self.fast_forward = True  # Defaults to fast mode so it processes quickly
        
        # Data history list
        self.history_log = []
        
        self.reset_run()

    def reset_run(self):
        self.avatars = []
        open_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if self.grid[x, y] == 0]
        
        spawn_spots = random.sample(open_positions, min(50, len(open_positions)))
        for spot in spawn_spots:
            self.avatars.append(Avatar(spot))
            
        self.predator_pos = random.choice([pos for pos in open_positions if pos not in spawn_spots])
        
        self.deaths = 0
        self.steps = 0
        self.fusions_count = 0
        self.predator_killed = False
        self.kill_lines = []  
        self.total_fusions_score = 0
        
        # Metrics to log *during* this run
        self.run_avatar_distances = []
        self.run_predator_distances = []

    def get_perception(self, pos):
        inputs = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        
        for dx, dy in directions:
            wall_dist = 5.0
            avatar_dist = 5.0
            predator_dist = 5.0
            
            for step in range(1, 6):
                nx, ny = pos[0] + dx * step, pos[1] + dy * step
                if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                    wall_dist = min(wall_dist, step)
                    break
                if self.grid[nx, ny] == 1:
                    wall_dist = min(wall_dist, step)
                    break
                for av in self.avatars:
                    if av.pos == (nx, ny):
                        avatar_dist = min(avatar_dist, step)
                if self.predator_pos == (nx, ny):
                    predator_dist = min(predator_dist, step)
                    
            inputs.append(1.0 / wall_dist)
            inputs.append(1.0 / avatar_dist if avatar_dist < 5.0 else 0.0)
            inputs.append(1.0 / predator_dist if predator_dist < 5.0 else 0.0)
            
        return np.array(inputs).reshape(1, -1)

    def update(self):
        self.steps += 1
        self.kill_lines = []
        
        # --- Track distances at this step ---
        if len(self.avatars) > 1:
            step_dists = []
            for av1 in self.avatars:
                min_d = min(abs(av1.pos[0] - av2.pos[0]) + abs(av1.pos[1] - av2.pos[1]) 
                            for av2 in self.avatars if av1 != av2)
                step_dists.append(min_d)
            self.run_avatar_distances.append(float(np.mean(step_dists)))
            
        if len(self.avatars) > 0 and not self.predator_killed:
            p_dists = [abs(av.pos[0] - self.predator_pos[0]) + abs(av.pos[1] - self.predator_pos[1]) for av in self.avatars]
            self.run_predator_distances.append(float(np.mean(p_dists)))
        
        # --- 1. Move Avatars ---
        for av in self.avatars:
            inputs = self.get_perception(av.pos)
            q_values = self.candidate_mind.forward(inputs)[0]
            
            if random.random() < 0.05:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_values)
                
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
            dx, dy = directions[action]
            new_pos = (av.pos[0] + dx, av.pos[1] + dy)
            
            if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE:
                if self.grid[new_pos[0], new_pos[1]] == 0:
                    av.pos = new_pos

        # --- 2. Fusion Mechanics ---
        pos_map = {}
        for av in self.avatars:
            pos_map.setdefault(av.pos, []).append(av)
            
        new_avatars_list = []
        for pos, avs_at_pos in pos_map.items():
            if len(avs_at_pos) > 1:
                surviving_avatar = avs_at_pos[0]
                total_size = sum(av.size for av in avs_at_pos)
                surviving_avatar.size = total_size
                new_avatars_list.append(surviving_avatar)
                self.fusions_count += (len(avs_at_pos) - 1)
                self.total_fusions_score += (len(avs_at_pos) - 1) * 200
            else:
                new_avatars_list.append(avs_at_pos[0])
        self.avatars = new_avatars_list

        # --- 3. Move Predator ---
        if self.steps % 2 == 0 and len(self.avatars) > 0 and not self.predator_killed:
            nearest_av = min(self.avatars, key=lambda a: abs(a.pos[0]-self.predator_pos[0]) + abs(a.pos[1]-self.predator_pos[1]))
            queue = [[self.predator_pos]]
            visited = {self.predator_pos}
            next_step = self.predator_pos
            
            while queue:
                path = queue.pop(0)
                curr = path[-1]
                if curr == nearest_av.pos:
                    if len(path) > 1:
                        next_step = path[1]
                    break
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = (curr[0] + dx, curr[1] + dy)
                    if 0 <= nxt[0] < GRID_SIZE and 0 <= nxt[1] < GRID_SIZE:
                        if self.grid[nxt[0], nxt[1]] == 0 and nxt not in visited:
                            visited.add(nxt)
                            queue.append(path + [nxt])
            self.predator_pos = next_step

        # --- 4. Sight/Laser Check ---
        if not self.predator_killed:
            px, py = self.predator_pos
            for av in list(self.avatars):
                ax, ay = av.pos
                if px == ax or py == ay:
                    distance = abs(px - ax) + abs(py - ay)
                    if distance <= 4:
                        blocked = False
                        if px == ax:
                            for y_step in range(min(py, ay) + 1, max(py, ay)):
                                if self.grid[px, y_step] == 1:
                                    blocked = True
                                    break
                        else:
                            for x_step in range(min(px, ax) + 1, max(px, ax)):
                                if self.grid[x_step, py] == 1:
                                    blocked = True
                                    break
                                    
                        if not blocked:
                            if av.size <= 25:
                                self.avatars.remove(av)
                                self.deaths += 1
                                self.kill_lines.append((self.predator_pos, av.pos))

        # --- 5. Collision Check ---
        if not self.predator_killed:
            for av in list(self.avatars):
                if abs(av.pos[0] - self.predator_pos[0]) + abs(av.pos[1] - self.predator_pos[1]) <= 1:
                    if av.size > 25:
                        self.predator_killed = True
                    else:
                        if av in self.avatars:
                            self.avatars.remove(av)
                            self.deaths += 1

        # --- 6. End of Run Check ---
        run_ended = False
        if self.deaths >= 25:
            run_ended = True
        elif len(self.avatars) <= 1 and self.fusions_count > 0:
            run_ended = True
        elif self.predator_killed:
            run_ended = True
        elif self.steps >= 1000:
            run_ended = True
            
        if run_ended:
            self.evaluate_generation()

    def evaluate_generation(self):
        fitness = (self.total_fusions_score) + (self.steps * 1.5) - (self.deaths * 150)
        if self.predator_killed:
            fitness += 8000
            
        # Log generation details
        avg_avatar_d = round(float(np.mean(self.run_avatar_distances)), 2) if self.run_avatar_distances else 0.0
        avg_predator_d = round(float(np.mean(self.run_predator_distances)), 2) if self.run_predator_distances else 0.0
        
        gen_data = {
            "generation": self.generation,
            "fitness": round(float(fitness), 1),
            "steps": self.steps,
            "fusions": self.fusions_count,
            "deaths": self.deaths,
            "predator_killed": self.predator_killed,
            "avg_distance_between_avatars": avg_avatar_d,
            "avg_distance_to_predator": avg_predator_d,
            "neural_attention": self.candidate_mind.get_attention_profile()
        }
        self.history_log.append(gen_data)
        
        # Print progress to console
        print(f"[GEN {self.generation:2d}] Logged: Fusions={self.fusions_count}, Deaths={self.deaths}, PredKilled={self.predator_killed}")

        if fitness > self.best_fitness or self.generation == 1:
            self.best_fitness = fitness
            self.best_mind = self.candidate_mind
            
        # --- IF PREDATOR WAS KILLED, EXPORT LOG AND EXIT ---
        if self.predator_killed:
            print("\n==========================================")
            print("VICTORY! The predator has been killed.")
            print("Saving data to 'simulation_log.json' and closing...")
            print("==========================================")
            
            with open("simulation_log.json", "w") as f:
                json.dump(self.history_log, f, indent=4)
                
            pygame.quit()
            sys.exit()
            
        self.generation += 1
        self.candidate_mind = self.best_mind.mutate(rate=0.08)
        self.reset_run()

    def draw(self):
        self.screen.fill(COLOR_BG)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[x, y] == 1:
                    pygame.draw.rect(self.screen, COLOR_WALL, rect)
                    pygame.draw.rect(self.screen, COLOR_BG, rect, 1)
                else:
                    pygame.draw.rect(self.screen, COLOR_PATH, rect)
                    pygame.draw.rect(self.screen, (35, 35, 45), rect, 1)
                    
        for p_pos, a_pos in self.kill_lines:
            p_center = (p_pos[1] * CELL_SIZE + CELL_SIZE//2, p_pos[0] * CELL_SIZE + CELL_SIZE//2)
            a_center = (a_pos[1] * CELL_SIZE + CELL_SIZE//2, a_pos[0] * CELL_SIZE + CELL_SIZE//2)
            pygame.draw.line(self.screen, COLOR_LASER, p_center, a_center, 4)
            
        if not self.predator_killed:
            px, py = self.predator_pos
            center = (py * CELL_SIZE + CELL_SIZE // 2, px * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, (231, 76, 60, 100), center, CELL_SIZE // 2 - 1)
            pygame.draw.circle(self.screen, COLOR_PREDATOR, center, CELL_SIZE // 3)
            
        for av in self.avatars:
            ax, ay = av.pos
            center = (ay * CELL_SIZE + CELL_SIZE // 2, ax * CELL_SIZE + CELL_SIZE // 2)
            radius = min(CELL_SIZE // 2 - 2, 6 + av.size * 1)
            color = COLOR_FUSED if av.size > 1 else COLOR_AVATAR
            if av.size > 25:
                pygame.draw.circle(self.screen, (241, 196, 15), center, radius + 2)
            pygame.draw.circle(self.screen, color, center, radius)
            if av.size > 1:
                sz_txt = self.font.render(str(av.size), True, COLOR_TEXT)
                self.screen.blit(sz_txt, (center[0] - sz_txt.get_width()//2, center[1] - sz_txt.get_height()//2))

        ui_top = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(self.screen, (10, 10, 15), (0, ui_top, WIDTH, UI_HEIGHT))
        
        col1, col2, col3 = 20, 240, 480
        gen_txt = self.font_bold.render(f"Generation: {self.generation}", True, (52, 152, 219))
        self.screen.blit(gen_txt, (col1, ui_top + 15))
        
        mode_txt_val = "FAST FORWARD" if self.fast_forward else "NORMAL SPEED"
        mode_color = (230, 126, 34) if self.fast_forward else (46, 204, 113)
        mode_txt = self.font_bold.render(f"Mode: {mode_txt_val}", True, mode_color)
        self.screen.blit(mode_txt, (col1, ui_top + 45))
        
        ctrl_txt = self.font.render("Press [SPACE] to toggle Fast training", True, (127, 140, 141))
        self.screen.blit(ctrl_txt, (col1, ui_top + 75))
        
        active_txt = self.font.render(f"Active Avatars: {len(self.avatars)}", True, COLOR_TEXT)
        self.screen.blit(active_txt, (col2, ui_top + 15))
        
        deaths_txt = self.font.render(f"Deaths: {self.deaths} / 25", True, COLOR_TEXT)
        self.screen.blit(deaths_txt, (col2, ui_top + 45))
        
        pred_txt = self.font.render(f"Predator: {'DEAD' if self.predator_killed else 'ALIVE'}", True, COLOR_TEXT)
        self.screen.blit(pred_txt, (col2, ui_top + 75))
        
        steps_txt = self.font.render(f"Steps: {self.steps} / 1000", True, COLOR_TEXT)
        self.screen.blit(steps_txt, (col3, ui_top + 15))
        
        best_fit = self.best_fitness if self.best_fitness != -999999 else 0
        fit_txt = self.font.render(f"Best Score: {best_fit:.0f}", True, (241, 196, 15))
        self.screen.blit(fit_txt, (col3, ui_top + 45))
        
        fused_cnt_txt = self.font.render(f"Fusions: {self.fusions_count}", True, COLOR_TEXT)
        self.screen.blit(fused_cnt_txt, (col3, ui_top + 75))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            fps = FPS_FAST if self.fast_forward else FPS_NORMAL
            self.clock.tick(fps)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.fast_forward = not self.fast_forward
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                        
            self.update()
            if not self.fast_forward or self.steps % 10 == 0:
                self.draw()

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    sim = Simulation()
    sim.run()