"""
Neural Hive vs Co-Evolving Predators
=====================================
v2 of the maze simulation. Major changes from v1:

  1. MAZE:      procedurally generated (randomized-DFS perfect maze + a loop-adding
                pass so it isn't just single-path corridors) instead of a hardcoded grid.
  2. LEARNING:  population-based genetic algorithm (many competing brains, not one
                shared brain) + "NEAT-lite" structural mutation -- hidden layer size
                can grow/shrink over generations, so network complexity itself evolves.
                This is a simplification of real NEAT (no innovation numbers / historical
                gene tracking, no per-connection topology) but gives you real topology
                evolution without the crossover-alignment complexity of full NEAT.
  3. HIVE:      each avatar is piloted by one genome out of a *population* of swarm
                brains (round-robin assigned), so a "match" has real behavioral
                diversity, not 50 clones. Avatars also deposit/sense a pheromone field
                (stigmergic signal), which is itself an evolved output of the network --
                so coordination is learned, not hardcoded.
  4. PREDATOR:  no longer scripted-with-BFS-only. A population of predator brains
                co-evolves against the swarm every generation -- a genuine arms race.
                Predators also sense the pheromone field, so the swarm evolving to
                "go quiet" vs. "stay coordinated" is a real trade-off that shows up
                in the logs.
  5. LOGGING:   every generation logs population-level statistics for both species
                (best/avg/worst fitness, network complexity, pheromone usage) so the
                arms race is fully documented, not just win/loss.

Run modes:
    python simulation_v2.py                     # windowed, real-time visualization
    python simulation_v2.py --headless --gens 300 --out run_log.json
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pygame

# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
GRID_SIZE = 25                  # must be odd for the maze generator
CELL_SIZE = 26
UI_HEIGHT = 150
WIDTH = GRID_SIZE * CELL_SIZE
HEIGHT = (GRID_SIZE * CELL_SIZE) + UI_HEIGHT

FPS_NORMAL = 12
FPS_FAST = 0                    # 0 = uncapped

N_AVATARS = 50
SWARM_POP_SIZE = 10             # 5 avatars piloted per swarm genome
N_PREDATORS = 5
PRED_POP_SIZE = 5               # 1 predator piloted per predator genome

MAX_STEPS = 10000
DEATH_LIMIT = 25
MAZE_REGEN_EVERY = 15           # regenerate maze every N generations (generalization)
LOOP_FACTOR = 0.10              # fraction of walls knocked down to add cycles/rooms

WEIGHT_MUTATE_RATE = 0.12
STRUCTURAL_MUTATE_PROB = 0.08   # chance a child's hidden layer grows/shrinks by 1
MIN_HIDDEN = 6
MAX_HIDDEN = 32
ELITE_KEEP = 2                  # top-N genomes carried over unmutated each generation

PHEROMONE_DECAY = 0.93
FUSION_SIZE_THRESHOLD = 25      # size needed to survive a laser hit / kill a predator

# Colors
COLOR_BG = (18, 18, 24)
COLOR_WALL = (44, 44, 56)
COLOR_PATH = (28, 28, 36)
COLOR_AVATAR = (41, 128, 185)
COLOR_FUSED = (155, 89, 182)
COLOR_PREDATOR = (231, 76, 60)
COLOR_TEXT = (236, 240, 241)
COLOR_LASER = (231, 76, 60)
COLOR_PHEROMONE = (46, 204, 113)

AVATAR_INPUT_SIZE = 18   # 4 dirs * (wall, avatar_sight, predator_sight, pheromone) + own_size + local_predator_count
AVATAR_OUTPUT_SIZE = 5   # 4 movement + 1 pheromone deposit strength
PRED_INPUT_SIZE = 16     # 4 dirs * (wall, pheromone, avatar_sight, other_predator_dist)
PRED_OUTPUT_SIZE = 4     # movement


# ----------------------------------------------------------------------------------
# Maze generation: randomized-DFS perfect maze, then punch extra openings for loops
# ----------------------------------------------------------------------------------
def generate_maze(size=GRID_SIZE, loop_factor=LOOP_FACTOR, rng=None):
    rng = rng or random
    if size % 2 == 0:
        size += 1
    grid = np.ones((size, size))
    start = (1, 1)
    grid[start] = 0
    stack = [start]
    while stack:
        x, y = stack[-1]
        neighbors = []
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < size - 1 and 0 < ny < size - 1 and grid[nx, ny] == 1:
                neighbors.append((nx, ny, dx, dy))
        if neighbors:
            nx, ny, dx, dy = rng.choice(neighbors)
            grid[x + dx // 2, y + dy // 2] = 0
            grid[nx, ny] = 0
            stack.append((nx, ny))
        else:
            stack.pop()

    # Loop-adding pass: knock down some interior walls so the maze has cycles/rooms
    # instead of being a single-path labyrinth (which would strangle 50+5 agents).
    interior_walls = [(x, y) for x in range(1, size - 1) for y in range(1, size - 1)
                       if grid[x, y] == 1]
    n_knock = int(len(interior_walls) * loop_factor)
    for x, y in rng.sample(interior_walls, min(n_knock, len(interior_walls))):
        grid[x, y] = 0

    return grid


# ----------------------------------------------------------------------------------
# Genome: variable-topology feedforward net with weight + structural mutation
# ----------------------------------------------------------------------------------
class Genome:
    def __init__(self, input_size, output_size, hidden_size=16, w1=None, b1=None, w2=None, b2=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.W1 = w1 if w1 is not None else np.random.randn(input_size, hidden_size) * 0.3
        self.b1 = b1 if b1 is not None else np.zeros((1, hidden_size))
        self.W2 = w2 if w2 is not None else np.random.randn(hidden_size, output_size) * 0.3
        self.b2 = b2 if b2 is not None else np.zeros((1, output_size))

    def forward(self, x):
        h = np.tanh(np.dot(x, self.W1) + self.b1)
        out = np.dot(h, self.W2) + self.b2
        return out

    def copy(self):
        return Genome(self.input_size, self.output_size, self.hidden_size,
                       self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy())

    def _resized_to(self, new_hidden):
        """Return copies of this genome's matrices resized to new_hidden, padding
        new neurons with small random weights or trimming extras."""
        w1, b1, w2 = self.W1, self.b1, self.W2
        cur = self.hidden_size
        if new_hidden == cur:
            return w1.copy(), b1.copy(), w2.copy()
        if new_hidden < cur:
            return w1[:, :new_hidden].copy(), b1[:, :new_hidden].copy(), w2[:new_hidden, :].copy()
        # growing: pad extra columns/rows with small random values
        extra = new_hidden - cur
        w1_pad = np.hstack([w1, np.random.randn(self.input_size, extra) * 0.1])
        b1_pad = np.hstack([b1, np.zeros((1, extra))])
        w2_pad = np.vstack([w2, np.random.randn(extra, self.output_size) * 0.1])
        return w1_pad, b1_pad, w2_pad

    def mutate(self, weight_rate=WEIGHT_MUTATE_RATE, structural_prob=STRUCTURAL_MUTATE_PROB):
        new_hidden = self.hidden_size
        if random.random() < structural_prob:
            delta = random.choice([-1, 1])
            new_hidden = int(np.clip(self.hidden_size + delta, MIN_HIDDEN, MAX_HIDDEN))
        w1, b1, w2 = self._resized_to(new_hidden)
        w1 = w1 + np.random.randn(*w1.shape) * weight_rate
        b1 = b1 + np.random.randn(*b1.shape) * weight_rate
        w2 = w2 + np.random.randn(*w2.shape) * weight_rate
        b2 = self.b2.copy() + np.random.randn(*self.b2.shape) * weight_rate
        return Genome(self.input_size, self.output_size, new_hidden, w1, b1, w2, b2)

    @staticmethod
    def crossover(parent_a, parent_b):
        """Uniform crossover. If hidden sizes differ, the child inherits parent_a's
        topology and parent_b is resized to match before mixing."""
        target_hidden = parent_a.hidden_size
        b_w1, b_b1, b_w2 = parent_b._resized_to(target_hidden)

        def mix(m1, m2):
            mask = np.random.rand(*m1.shape) < 0.5
            return np.where(mask, m1, m2)

        w1 = mix(parent_a.W1, b_w1)
        b1 = mix(parent_a.b1, b_b1)
        w2 = mix(parent_a.W2, b_w2)
        b2 = mix(parent_a.b2, parent_b.b2)
        return Genome(parent_a.input_size, parent_a.output_size, target_hidden, w1, b1, w2, b2)

    def attention_profile(self, group_indices):
        """group_indices: dict name -> list of input-row indices. Returns mean |W1|
        for each group -- what the network is prioritizing mathematically."""
        return {name: round(float(np.mean(np.abs(self.W1[idx, :]))), 4)
                for name, idx in group_indices.items()}


# ----------------------------------------------------------------------------------
# Population: a set of competing/co-operating genomes for one species
# ----------------------------------------------------------------------------------
class Population:
    def __init__(self, size, input_size, output_size, hidden_size=16):
        self.genomes = [Genome(input_size, output_size, hidden_size) for _ in range(size)]
        self.fitness = [0.0] * size
        self.input_size = input_size
        self.output_size = output_size

    def reset_fitness(self):
        self.fitness = [0.0] * len(self.genomes)

    def stats(self):
        f = self.fitness
        return {
            "best_fitness": round(max(f), 2),
            "avg_fitness": round(float(np.mean(f)), 2),
            "worst_fitness": round(min(f), 2),
            "avg_complexity": round(float(np.mean([g.hidden_size for g in self.genomes])), 2),
            "best_genome_complexity": self.genomes[int(np.argmax(f))].hidden_size,
        }

    def best_genome(self):
        return self.genomes[int(np.argmax(self.fitness))]

    def evolve(self):
        ranked = sorted(range(len(self.genomes)), key=lambda i: self.fitness[i], reverse=True)
        new_genomes = [self.genomes[i].copy() for i in ranked[:ELITE_KEEP]]  # elitism

        def tournament(k=3):
            contenders = random.sample(range(len(self.genomes)), min(k, len(self.genomes)))
            best = max(contenders, key=lambda i: self.fitness[i])
            return self.genomes[best]

        while len(new_genomes) < len(self.genomes):
            parent_a = tournament()
            parent_b = tournament()
            child = Genome.crossover(parent_a, parent_b)
            child = child.mutate()
            new_genomes.append(child)

        self.genomes = new_genomes
        self.reset_fitness()


# ----------------------------------------------------------------------------------
# Lightweight agent state
# ----------------------------------------------------------------------------------
class Avatar:
    def __init__(self, pos, genome_idx):
        self.pos = pos
        self.size = 1
        self.genome_idx = genome_idx  # which swarm genome piloted this body


class Predator:
    def __init__(self, pos, genome_idx):
        self.pos = pos
        self.genome_idx = genome_idx
        self.alive = True


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


# ----------------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------------
class Simulation:
    def __init__(self, headless=False, max_generations=300, maze_seed=None):
        self.headless = headless
        self.max_generations = max_generations
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy" if headless else "")
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Neural Hive vs Co-Evolving Predators")
        self.clock = pygame.time.Clock()
        if not headless:
            self.font = pygame.font.SysFont("Arial", 15)
            self.font_bold = pygame.font.SysFont("Arial", 17, bold=True)

        self.rng = random.Random(maze_seed)
        self.maze_seed = maze_seed if maze_seed is not None else random.randint(0, 10_000_000)
        self.grid = generate_maze(GRID_SIZE, LOOP_FACTOR, random.Random(self.maze_seed))

        self.swarm_pop = Population(SWARM_POP_SIZE, AVATAR_INPUT_SIZE, AVATAR_OUTPUT_SIZE, hidden_size=16)
        self.pred_pop = Population(PRED_POP_SIZE, PRED_INPUT_SIZE, PRED_OUTPUT_SIZE, hidden_size=12)

        self.generation = 1
        self.fast_forward = True
        self.history_log = []

        self.reset_match()

    # -------------------------------------------------------------- match setup
    def reset_match(self):
        if self.generation % MAZE_REGEN_EVERY == 0:
            self.maze_seed = random.randint(0, 10_000_000)
            self.grid = generate_maze(GRID_SIZE, LOOP_FACTOR, random.Random(self.maze_seed))

        open_positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE) if self.grid[x, y] == 0]

        self.avatars = []
        spawn_spots = self.rng.sample(open_positions, min(N_AVATARS, len(open_positions)))
        for i, spot in enumerate(spawn_spots):
            self.avatars.append(Avatar(spot, genome_idx=i % SWARM_POP_SIZE))

        remaining = [p for p in open_positions if p not in spawn_spots]
        pred_spots = self.rng.sample(remaining, min(N_PREDATORS, len(remaining)))
        self.predators = [Predator(spot, genome_idx=i % PRED_POP_SIZE) for i, spot in enumerate(pred_spots)]

        self.pheromone = np.zeros((GRID_SIZE, GRID_SIZE))

        self.deaths = 0
        self.steps = 0
        self.fusions_count = 0
        self.total_fusions_score = 0
        self.kill_lines = []
        self.pheromone_output_samples = []

        # per-genome credit accumulators for this match
        self.swarm_credit = [0.0] * SWARM_POP_SIZE
        self.pred_credit = [0.0] * PRED_POP_SIZE

    # -------------------------------------------------------------- perception
    def _ray(self, pos, dx, dy, max_range=5):
        wall_dist = float(max_range)
        for step in range(1, max_range + 1):
            nx, ny = pos[0] + dx * step, pos[1] + dy * step
            if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE or self.grid[nx, ny] == 1:
                wall_dist = step
                break
        return wall_dist

    def get_avatar_perception(self, av):
        inputs = []
        for dx, dy in DIRECTIONS:
            wall_dist = 5.0
            avatar_dist = 5.0
            predator_dist = 5.0
            for step in range(1, 6):
                nx, ny = av.pos[0] + dx * step, av.pos[1] + dy * step
                if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
                    wall_dist = min(wall_dist, step)
                    break
                if self.grid[nx, ny] == 1:
                    wall_dist = min(wall_dist, step)
                    break
                for other in self.avatars:
                    if other is not av and other.pos == (nx, ny):
                        avatar_dist = min(avatar_dist, step)
                for pred in self.predators:
                    if pred.alive and pred.pos == (nx, ny):
                        predator_dist = min(predator_dist, step)
            inputs.append(1.0 / wall_dist)
            inputs.append(1.0 / avatar_dist if avatar_dist < 5.0 else 0.0)
            inputs.append(1.0 / predator_dist if predator_dist < 5.0 else 0.0)
            # local pheromone gradient: sample the neighboring cell one step away
            nx, ny = av.pos[0] + dx, av.pos[1] + dy
            phero = self.pheromone[nx, ny] if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE else 0.0
            inputs.append(float(phero))

        own_size_norm = min(av.size / 50.0, 1.0)
        nearby_predators = sum(
            1 for p in self.predators
            if p.alive and abs(p.pos[0] - av.pos[0]) + abs(p.pos[1] - av.pos[1]) <= 5
        ) / max(N_PREDATORS, 1)
        inputs.append(own_size_norm)
        inputs.append(nearby_predators)
        return np.array(inputs).reshape(1, -1)

    def get_predator_perception(self, pred):
        inputs = []
        for dx, dy in DIRECTIONS:
            wall_dist = self._ray(pred.pos, dx, dy)
            nx, ny = pred.pos[0] + dx, pred.pos[1] + dy
            phero = self.pheromone[nx, ny] if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE else 0.0

            avatar_dist = 5.0
            other_pred_dist = 5.0
            for step in range(1, 6):
                nx2, ny2 = pred.pos[0] + dx * step, pred.pos[1] + dy * step
                if nx2 < 0 or nx2 >= GRID_SIZE or ny2 < 0 or ny2 >= GRID_SIZE or self.grid[nx2, ny2] == 1:
                    break
                for av in self.avatars:
                    if av.pos == (nx2, ny2):
                        avatar_dist = min(avatar_dist, step)
                for other in self.predators:
                    if other is not pred and other.alive and other.pos == (nx2, ny2):
                        other_pred_dist = min(other_pred_dist, step)

            inputs.append(1.0 / wall_dist)
            inputs.append(float(phero))
            inputs.append(1.0 / avatar_dist if avatar_dist < 5.0 else 0.0)
            inputs.append(1.0 / other_pred_dist if other_pred_dist < 5.0 else 0.0)
        return np.array(inputs).reshape(1, -1)

    # -------------------------------------------------------------- step
    def update(self):
        self.steps += 1
        self.kill_lines = []

        # 1. Move avatars + deposit pheromone (learned output, not hardcoded)
        for av in self.avatars:
            genome = self.swarm_pop.genomes[av.genome_idx]
            inputs = self.get_avatar_perception(av)
            out = genome.forward(inputs)[0]
            move_logits, phero_signal = out[:4], out[4]

            if random.random() < 0.05:
                action = random.randint(0, 3)
            else:
                action = int(np.argmax(move_logits))
            dx, dy = DIRECTIONS[action]
            new_pos = (av.pos[0] + dx, av.pos[1] + dy)
            if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and self.grid[new_pos] == 0:
                av.pos = new_pos

            deposit = 1.0 / (1.0 + np.exp(-phero_signal))  # sigmoid -> [0,1]
            self.pheromone[av.pos] = min(1.0, self.pheromone[av.pos] + deposit * 0.5)
            self.pheromone_output_samples.append(float(deposit))

        # 2. Fusion mechanics (credit goes to the surviving avatar's genome)
        pos_map = {}
        for av in self.avatars:
            pos_map.setdefault(av.pos, []).append(av)
        new_avatars = []
        for pos, group in pos_map.items():
            if len(group) > 1:
                survivor = group[0]
                survivor.size = sum(a.size for a in group)
                new_avatars.append(survivor)
                n_fused = len(group) - 1
                self.fusions_count += n_fused
                self.total_fusions_score += n_fused * 200
                self.swarm_credit[survivor.genome_idx] += n_fused * 200
            else:
                new_avatars.append(group[0])
        self.avatars = new_avatars

        # 3. Move predators (their own evolved brains, not BFS)
        for pred in self.predators:
            if not pred.alive:
                continue
            genome = self.pred_pop.genomes[pred.genome_idx]
            inputs = self.get_predator_perception(pred)
            out = genome.forward(inputs)[0]
            action = int(np.argmax(out))
            dx, dy = DIRECTIONS[action]
            new_pos = (pred.pos[0] + dx, pred.pos[1] + dy)
            if 0 <= new_pos[0] < GRID_SIZE and 0 <= new_pos[1] < GRID_SIZE and self.grid[new_pos] == 0:
                pred.pos = new_pos
            # small per-step survival/pursuit shaping reward
            if self.avatars:
                nearest_d = min(abs(pred.pos[0] - a.pos[0]) + abs(pred.pos[1] - a.pos[1]) for a in self.avatars)
                self.pred_credit[pred.genome_idx] += max(0.0, (10 - nearest_d)) * 0.2

        # 4. Sight/laser check (predator kills small avatars at range if line-of-sight clear)
        for pred in self.predators:
            if not pred.alive:
                continue
            px, py = pred.pos
            for av in list(self.avatars):
                ax, ay = av.pos
                if px == ax or py == ay:
                    distance = abs(px - ax) + abs(py - ay)
                    if 0 < distance <= 4:
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
                        if not blocked and av.size <= FUSION_SIZE_THRESHOLD:
                            self.avatars.remove(av)
                            self.deaths += 1
                            self.kill_lines.append((pred.pos, av.pos))
                            self.pred_credit[pred.genome_idx] += 300

        # 5. Collision check (giant avatar kills predator on contact, else predator eats it)
        for pred in self.predators:
            if not pred.alive:
                continue
            for av in list(self.avatars):
                if abs(av.pos[0] - pred.pos[0]) + abs(av.pos[1] - pred.pos[1]) <= 1:
                    if av.size > FUSION_SIZE_THRESHOLD:
                        pred.alive = False
                        self.swarm_credit[av.genome_idx] += 2000
                        self.pred_credit[pred.genome_idx] -= 500
                    elif av in self.avatars:
                        self.avatars.remove(av)
                        self.deaths += 1
                        self.pred_credit[pred.genome_idx] += 300

        # 6. Pheromone decay
        self.pheromone *= PHEROMONE_DECAY

        # 7. End-of-match check
        predators_alive = sum(1 for p in self.predators if p.alive)
        run_ended = False
        if self.deaths >= DEATH_LIMIT:
            run_ended = True
        elif len(self.avatars) <= 1 and self.fusions_count > 0:
            run_ended = True
        elif predators_alive == 0:
            run_ended = True
        elif self.steps >= MAX_STEPS:
            run_ended = True

        if run_ended:
            self.evaluate_generation()

    # -------------------------------------------------------------- generation wrap-up
    def evaluate_generation(self):
        predators_alive = sum(1 for p in self.predators if p.alive)
        all_predators_killed = predators_alive == 0

        # survival-time shaping for both sides, on top of event-based credit above
        for av in self.avatars:
            self.swarm_credit[av.genome_idx] += self.steps * 0.3
        for i in range(PRED_POP_SIZE):
            if predators_alive > 0:
                self.pred_credit[i] += self.steps * 0.1  # predators that are still hunting get a small bonus

        for i in range(SWARM_POP_SIZE):
            self.swarm_pop.fitness[i] += self.swarm_credit[i]
        for i in range(PRED_POP_SIZE):
            self.pred_pop.fitness[i] += self.pred_credit[i]

        swarm_stats = self.swarm_pop.stats()
        pred_stats = self.pred_pop.stats()
        avg_phero = round(float(np.mean(self.pheromone_output_samples)), 4) if self.pheromone_output_samples else 0.0

        gen_data = {
            "generation": self.generation,
            "steps": self.steps,
            "fusions": self.fusions_count,
            "deaths": self.deaths,
            "predators_alive_end": predators_alive,
            "all_predators_killed": all_predators_killed,
            "maze_seed": self.maze_seed,
            "swarm": {
                "best_fitness": swarm_stats["best_fitness"],
                "avg_fitness": swarm_stats["avg_fitness"],
                "worst_fitness": swarm_stats["worst_fitness"],
                "avg_hidden_complexity": swarm_stats["avg_complexity"],
                "best_genome_hidden_size": swarm_stats["best_genome_complexity"],
                "avg_pheromone_signal": avg_phero,
                "attention": self.swarm_pop.best_genome().attention_profile({
                    "wall": [0, 4, 8, 12],
                    "avatar_sight": [1, 5, 9, 13],
                    "predator_sight": [2, 6, 10, 14],
                    "pheromone": [3, 7, 11, 15],
                }),
            },
            "predator": {
                "best_fitness": pred_stats["best_fitness"],
                "avg_fitness": pred_stats["avg_fitness"],
                "worst_fitness": pred_stats["worst_fitness"],
                "avg_hidden_complexity": pred_stats["avg_complexity"],
                "best_genome_hidden_size": pred_stats["best_genome_complexity"],
                "attention": self.pred_pop.best_genome().attention_profile({
                    "wall": [0, 4, 8, 12],
                    "pheromone": [1, 5, 9, 13],
                    "avatar_sight": [2, 6, 10, 14],
                    "other_predator": [3, 7, 11, 15],
                }),
            },
            "arms_race_gap": round(swarm_stats["best_fitness"] - pred_stats["best_fitness"], 2),
        }
        self.history_log.append(gen_data)

        print(f"[GEN {self.generation:3d}] steps={self.steps:4d} fusions={self.fusions_count:2d} "
              f"deaths={self.deaths:2d} pred_alive={predators_alive}/{N_PREDATORS} "
              f"swarm_best={swarm_stats['best_fitness']:.0f} pred_best={pred_stats['best_fitness']:.0f}")

        self.swarm_pop.evolve()
        self.pred_pop.evolve()

        if self.generation >= self.max_generations:
            self.save_and_exit()

        self.generation += 1
        self.reset_match()

    def save_and_exit(self, path="simulation_log_v2.json"):
        print("\n==========================================")
        print(f"Training complete after {self.generation} generations.")
        print(f"Saving data to '{path}' and closing...")
        print("==========================================")
        with open(path, "w") as f:
            json.dump(self.history_log, f, indent=2)
        pygame.quit()
        sys.exit()

    # -------------------------------------------------------------- rendering
    def draw(self):
        self.screen.fill(COLOR_BG)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                if self.grid[x, y] == 1:
                    pygame.draw.rect(self.screen, COLOR_WALL, rect)
                else:
                    phero = self.pheromone[x, y]
                    if phero > 0.02:
                        c = tuple(int(COLOR_PATH[i] + (COLOR_PHEROMONE[i] - COLOR_PATH[i]) * min(phero, 1.0) * 0.6)
                                  for i in range(3))
                        pygame.draw.rect(self.screen, c, rect)
                    else:
                        pygame.draw.rect(self.screen, COLOR_PATH, rect)
                    pygame.draw.rect(self.screen, (35, 35, 45), rect, 1)

        for p_pos, a_pos in self.kill_lines:
            p_center = (p_pos[1] * CELL_SIZE + CELL_SIZE // 2, p_pos[0] * CELL_SIZE + CELL_SIZE // 2)
            a_center = (a_pos[1] * CELL_SIZE + CELL_SIZE // 2, a_pos[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.line(self.screen, COLOR_LASER, p_center, a_center, 3)

        for pred in self.predators:
            if not pred.alive:
                continue
            px, py = pred.pos
            center = (py * CELL_SIZE + CELL_SIZE // 2, px * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, COLOR_PREDATOR, center, CELL_SIZE // 3)

        for av in self.avatars:
            ax, ay = av.pos
            center = (ay * CELL_SIZE + CELL_SIZE // 2, ax * CELL_SIZE + CELL_SIZE // 2)
            radius = min(CELL_SIZE // 2 - 2, 5 + av.size)
            color = COLOR_FUSED if av.size > 1 else COLOR_AVATAR
            if av.size > FUSION_SIZE_THRESHOLD:
                pygame.draw.circle(self.screen, (241, 196, 15), center, radius + 2)
            pygame.draw.circle(self.screen, color, center, radius)

        ui_top = GRID_SIZE * CELL_SIZE
        pygame.draw.rect(self.screen, (10, 10, 15), (0, ui_top, WIDTH, UI_HEIGHT))
        col1, col2, col3 = 15, 230, 450
        lines = [
            (f"Gen: {self.generation}/{self.max_generations}", col1, 10, (52, 152, 219)),
            (f"Steps: {self.steps}/{MAX_STEPS}", col1, 32, COLOR_TEXT),
            (f"Mode: {'FAST' if self.fast_forward else 'NORMAL'}", col1, 54, (230, 126, 34)),
            (f"[SPACE] toggle speed", col1, 76, (127, 140, 141)),
            (f"Avatars: {len(self.avatars)}  Deaths: {self.deaths}/{DEATH_LIMIT}", col2, 10, COLOR_TEXT),
            (f"Predators alive: {sum(1 for p in self.predators if p.alive)}/{N_PREDATORS}", col2, 32, COLOR_TEXT),
            (f"Fusions: {self.fusions_count}", col2, 54, COLOR_TEXT),
            (f"Swarm best fit: {max(self.swarm_pop.fitness):.0f}", col3, 10, (241, 196, 15)),
            (f"Pred best fit: {max(self.pred_pop.fitness):.0f}", col3, 32, (231, 76, 60)),
            (f"Complexity S/P: {self.swarm_pop.stats()['avg_complexity']:.1f}/{self.pred_pop.stats()['avg_complexity']:.1f}",
             col3, 54, (149, 165, 166)),
        ]
        for text, x, y, color in lines:
            surf = self.font.render(text, True, color)
            self.screen.blit(surf, (x, ui_top + y))

        pygame.display.flip()

    # -------------------------------------------------------------- main loop
    def run(self):
        running = True
        while running:
            if not self.headless:
                fps = FPS_FAST if self.fast_forward else FPS_NORMAL
                self.clock.tick(fps)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.fast_forward = not self.fast_forward
                        elif event.key == pygame.K_ESCAPE:
                            self.save_and_exit()

            self.update()
            if not self.headless and (not self.fast_forward or self.steps % 5 == 0):
                self.draw()

        self.save_and_exit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="run without a window, as fast as possible")
    parser.add_argument("--gens", type=int, default=300, help="number of generations to train")
    parser.add_argument("--out", type=str, default="simulation_log_v2.json", help="output log path")
    parser.add_argument("--seed", type=int, default=None, help="maze seed for reproducibility")
    args = parser.parse_args()

    sim = Simulation(headless=args.headless, max_generations=args.gens, maze_seed=args.seed)
    sim.save_path = args.out
    # monkey-patch save path
    orig_save = sim.save_and_exit
    sim.save_and_exit = lambda path=args.out: orig_save(path)
    sim.run()


if __name__ == "__main__":
    main()
