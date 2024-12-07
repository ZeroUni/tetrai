import pygame
import random
import numpy as np
from pygame import Surface
from typing import Tuple

import threading
import multiprocessing

import torch
import cupy as cp
from numba import jit, cuda
from functools import lru_cache

# Define constants
SCREEN_SIZE = 512
SCREEN_WIDTH = SCREEN_SIZE # MAKE IT A SQUARE BABY
SCREEN_HEIGHT = SCREEN_SIZE
BLOCK_SIZE = 20
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BOARD_MARGIN = 20

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Define piece IDs
PIECE_IDS = {
    'I': 1,
    'J': 2,
    'L': 3,
    'O': 4,
    'S': 5,
    'T': 6,
    'Z': 7
}

# Define colors for each shape
COLORS = {
    1: (0, 255, 255),    # Cyan
    2: (0, 0, 255),      # Blue
    3: (255, 165, 0),    # Orange
    4: (255, 255, 0),    # Yellow
    5: (0, 255, 0),      # Green
    6: (128, 0, 128),    # Purple
    7: (255, 0, 0)       # Red
}

SECONDARY_COLORS = {
    1: (0, 200, 200),    # Cyan
    2: (0, 0, 200),      # Blue
    3: (200, 100, 0),    # Orange
    4: (200, 200, 0),    # Yellow
    5: (0, 200, 0),      # Green
    6: (100, 0, 100),    # Purple
    7: (200, 0, 0)       # Red
}

# Update shapes with rotation states (using SRS)
SHAPES = {
    'I': [
        [[0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 1, 1, 1],
         [0, 0, 0, 0]],
        [[0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0],
         [0, 1, 0, 0]]
    ],
    'J': [
        [[1, 0, 0],
         [1, 1, 1],
         [0, 0, 0]],
        [[0, 1, 1],
         [0, 1, 0],
         [0, 1, 0]],
        [[0, 0, 0],
         [1, 1, 1],
         [0, 0, 1]],
        [[0, 1, 0],
         [0, 1, 0],
         [1, 1, 0]]
    ],
    'L': [
        [[0, 0, 1],
         [1, 1, 1],
         [0, 0, 0]],
        [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 1]],
        [[0, 0, 0],
         [1, 1, 1],
         [1, 0, 0]],
        [[1, 1, 0],
         [0, 1, 0],
         [0, 1, 0]]
    ],
    'O': [
        [[1, 1],
         [1, 1]],
        [[1, 1],
         [1, 1]],
        [[1, 1],
         [1, 1]],
        [[1, 1],
         [1, 1]]
    ],
    'S': [
        [[0, 1, 1],
         [1, 1, 0],
         [0, 0, 0]],
        [[0, 1, 0],
         [0, 1, 1],
         [0, 0, 1]],
        [[0, 0, 0],
         [0, 1, 1],
         [1, 1, 0]],
        [[1, 0, 0],
         [1, 1, 0],
         [0, 1, 0]]
    ],
    'T': [
        [[0, 1, 0],
         [1, 1, 1],
         [0, 0, 0]],
        [[0, 1, 0],
         [0, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [1, 1, 1],
         [0, 1, 0]],
        [[0, 1, 0],
         [1, 1, 0],
         [0, 1, 0]]
    ],
    'Z': [
        [[1, 1, 0],
         [0, 1, 1],
         [0, 0, 0]],
        [[0, 0, 1],
         [0, 1, 1],
         [0, 1, 0]],
        [[0, 0, 0],
         [1, 1, 0],
         [0, 1, 1]],
        [[0, 1, 0],
         [1, 1, 0],
         [1, 0, 0]]
    ]
}

# Define wall kick data for SRS (Actual black magic what is this)
WALL_KICKS = {
    'I': {
        (0, 1): [(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
        (1, 0): [(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
        (1, 2): [(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],
        (2, 1): [(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
        (2, 3): [(0, 0), (+2, 0), (-1, 0), (+2, +1), (-1, -2)],
        (3, 2): [(0, 0), (-2, 0), (+1, 0), (-2, -1), (+1, +2)],
        (3, 0): [(0, 0), (+1, 0), (-2, 0), (+1, -2), (-2, +1)],
        (0, 3): [(0, 0), (-1, 0), (+2, 0), (-1, +2), (+2, -1)],
    },
    'J': {
        (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
    },
    'L': {
        (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
    },
    'S': {
        (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
    },
    'T': {
        (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
    },
    'Z': {
        (0, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (1, 0): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (1, 2): [(0, 0), (+1, 0), (+1, -1), (0, +2), (+1, +2)],
        (2, 1): [(0, 0), (-1, 0), (-1, +1), (0, -2), (-1, -2)],
        (2, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
        (3, 2): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (3, 0): [(0, 0), (-1, 0), (-1, -1), (0, +2), (-1, +2)],
        (0, 3): [(0, 0), (+1, 0), (+1, +1), (0, -2), (+1, -2)],
    },
    'O': {
        (0, 1): [(0, 0)], # No wall kicks for O piece
    }
}

class GameState:
    def __init__(self):
        self.board = torch.zeros((BOARD_HEIGHT, BOARD_WIDTH), 
                               dtype=torch.int8,
                               device='cuda')
        self.temp_board = torch.zeros_like(self.board)
        self.visited = torch.zeros_like(self.board, dtype=torch.bool)
        self.current_piece = self.get_new_piece()
        self.next_pieces = [self.get_new_piece() for _ in range(5)]
        self.hold_piece = None
        self.can_hold = True
        self.position = [0, BOARD_WIDTH // 2 - 2]
        self.rotation_index = 0
        self.score = 0
        self.lines_cleared = 0
        self.game_over = False
        self.lock_delay = 500  # Lock delay in milliseconds
        self.lock_timer = 0
        self.is_landing = False
        self.actions_per_piece = 0

        # Cache piece matrices as tensors
        self.piece_tensors = {
            piece_type: torch.tensor(shapes, device='cuda', dtype=torch.int8)
            for piece_type, shapes in SHAPES.items()
        }

    def get_new_piece(self):
        shape_type = random.choice(list(SHAPES.keys()))
        return {'type': shape_type, 'shape': SHAPES[shape_type]}

    def rotate_piece(self, direction):
        old_rotation = self.rotation_index
        old_position = self.position.copy()
        self.rotation_index = (self.rotation_index + direction) % 4
        rotated_shape = self.current_piece['shape'][self.rotation_index]

        rotation_transition = (old_rotation, self.rotation_index)
        kicks = WALL_KICKS[self.current_piece['type']].get(rotation_transition, [])

        for dx, dy in kicks:
            self.position[1] += dx
            self.position[0] += dy
            if self.valid_position(piece=rotated_shape):
                return  # Successful rotation with kick
            self.position = old_position.copy()

        # Revert rotation if all kicks fail
        self.rotation_index = old_rotation

    @staticmethod
    @jit(nopython=True, parallel=True)
    def count_holes_fast(board):
        """Optimized hole counting using Numba"""
        holes = 0
        for col in range(board.shape[1]):
            block_found = False
            for row in range(board.shape[0]):
                if board[row,col] != 0:
                    block_found = True
                elif block_found:
                    holes += 1
        return holes

    def hold_current_piece(self):
        if self.can_hold:
            if self.hold_piece is None:
                self.hold_piece = self.current_piece
                self.current_piece = self.next_pieces.pop(0)
                self.next_pieces.append(self.get_new_piece())
            else:
                self.hold_piece, self.current_piece = self.current_piece, self.hold_piece
            self.position = [0, BOARD_WIDTH // 2 - 2]
            self.rotation_index = 0
            self.can_hold = False

    @torch.jit.script
    def valid_position(self, piece=None, adj_x=0, adj_y=0):
        """Optimized collision detection with GPU tensors
        
        Args:
            piece: Optional piece matrix. Uses current piece if None
            adj_x: X position adjustment
            adj_y: Y position adjustment
            
        Returns:
            bool: True if position is valid
        """
        if piece is None:
            piece = self.current_piece['shape'][self.rotation_index]
        piece = torch.tensor(piece, device='cuda', dtype=torch.int8)
        
        # Calculate new position
        new_x = self.position[1] + adj_x 
        new_y = self.position[0] + adj_y

        # Bounds check
        if (new_x < 0 or 
            new_x + piece.shape[1] > BOARD_WIDTH or 
            new_y + piece.shape[0] > BOARD_HEIGHT):
            return False
            
        if new_y < 0:
            return True  # Allow piece to be partially above board
        
        # Check collision with board using GPU
        piece_area = self.board[new_y:new_y + piece.shape[0],
                            new_x:new_x + piece.shape[1]]
        return not torch.any(piece_area & piece)

    def lock_piece(self):
        self.can_hold = False
        shape = self.current_piece['shape'][self.rotation_index]
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.position[1] + x
                    new_y = self.position[0] + y
                    if new_y >= 0:
                        self.board[new_y][new_x] = PIECE_IDS[self.current_piece['type']]
        self.clear_lines()
        self.current_piece = self.next_pieces.pop(0)
        self.next_pieces.append(self.get_new_piece())
        curr_y = self.position[0]
        self.position = [0, BOARD_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2]
        self.rotation_index = 0
        if not self.valid_position():
            self.game_over = True
        # Up the score a bit based on the y level of the piece, e.g. 20 points for dropping a piece on the bottom
        self.score += curr_y
        self.can_hold = True
        self.is_landing = False
        self.lock_timer = 0
        self.actions_per_piece = 0

    @torch.jit.script
    def clear_lines(self):
        """Clear full lines using GPU operations
        
        Returns:
            int: Number of lines cleared
        """
        # Find full lines using GPU
        full_lines = torch.all(self.board != 0, dim=1)
        num_cleared = int(torch.sum(full_lines).item())
        
        if num_cleared == 0:
            return 0
            
        # Remove full lines
        keep_lines = ~full_lines
        self.board = self.board[keep_lines]
        
        # Add new empty lines on top
        empty_lines = torch.zeros((num_cleared, BOARD_WIDTH), 
                                dtype=torch.int8,
                                device='cuda')
        self.board = torch.cat([empty_lines, self.board])
        
        # Update score
        lines_before = self.lines_cleared
        self.lines_cleared += num_cleared
        
        # Score calculation
        match num_cleared:
            case 1: self.score += 40
            case 2: self.score += 500
            case 3: self.score += 1200
            case 4: self.score += 2800
        
        # Level up bonus
        if self.lines_cleared >= 10 * (lines_before // 10 + 1):
            self.score += 1000
            
        return num_cleared

class TetrisEnv:
    def __init__(self, render_queue=None, max_moves=-1, weights=None, manual=False):
        # Pre-allocate surfaces for rendering
        self.board_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.piece_surface = pygame.Surface((BLOCK_SIZE * 4, BLOCK_SIZE * 4))

        # Use GPU tensors for state processing
        self.state_processor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 3, padding=1),
            torch.nn.ReLU()
        ).cuda()

        if not pygame.get_init():
            pygame.init()
        
        self.manual = manual
        if self.manual:
            self.screen = pygame.display.set_mode((int(SCREEN_WIDTH), int(SCREEN_HEIGHT)))
        else:
            self.screen = None
        pygame.display.set_caption('TetrisEnv')
        self.clock = pygame.time.Clock()
        self.game_state = GameState()
        self.fall_time = 0
        self.fall_speed = 0.5  # Seconds per fall
        self.last_move_time = pygame.time.get_ticks()
        self.render_mode = False  # Set to False to run without rendering
        self.render_delay = 100  # Delay between renders in milliseconds
        self.last_render_time = pygame.time.get_ticks()

        self.lock = threading.Lock()
        self.render_queue = render_queue

        self.max_moves = max_moves
        self.move_count = 0

        self.weights = weights or {}

        # Calculate board display area
        self.board_display_width = BOARD_WIDTH * BLOCK_SIZE
        self.board_display_height = BOARD_HEIGHT * BLOCK_SIZE
        self.board_x = SCREEN_WIDTH // 2 - self.board_display_width // 2
        self.board_y = SCREEN_HEIGHT - self.board_display_height - BOARD_MARGIN

        # Side panel dimensions
        self.side_panel_x_right = self.board_x + self.board_display_width + 30
        self.side_panel_width_right = 4 * BLOCK_SIZE + 20
        self.side_panel_x_left = self.board_x - 20
        self.side_panel_width_left = self.board_x - 50


    def reset(self) -> np.ndarray:
        with self.lock:
            self.game_state = GameState()
            self.fall_time = 0
            self.last_move_time = pygame.time.get_ticks()
            self.move_count = 0
            return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        with self.lock:
            reward = 0
            done = False

            # Apply action
            self.apply_action(action)

            # Update game state
            self.update_game_state()

            # Calculate reward
            reward += self.calculate_reward()

            if self.game_state.game_over:
                done = True
                reward -= self.weights.get('game_over') or 1000  # Penalty for dying

            self.move_count += 1
            if 0 < self.max_moves <= self.move_count:
                done = True  # End episode due to max moves

            return self.get_state(), reward, done


    def apply_action(self, action: int):
        moved = False
        self.game_state.actions_per_piece += 1
        match action:
            case 0:  # Nothing
                self.game_state.actions_per_piece -= 1
                return
            case 1:  # Rotate Right
                self.game_state.rotate_piece(direction=1)
                moved = True
            case 2:  # Rotate Left
                self.game_state.rotate_piece(direction=-1)
                moved = True
            case 3:  # Move Down
                if self.game_state.valid_position(adj_y=+1):
                    self.game_state.position[0] += 1
                    moved = True
                self.game_state.actions_per_piece -= 1
            case 4:  # Move Left
                if self.game_state.valid_position(adj_x=-1):
                    self.game_state.position[1] -= 1
                    moved = True
            case 5:  # Move Right
                if self.game_state.valid_position(adj_x=1):
                    self.game_state.position[1] += 1
                    moved = True
            case 6:  # Hard Drop
                while self.game_state.valid_position(adj_y=1):
                    self.game_state.position[0] += 1
                self.game_state.lock_piece()
                self.game_state.is_landing = False
                self.game_state.lock_timer = 0
                self.game_state.actions_per_piece -= 1
            case 7:  # Hold
                self.game_state.hold_current_piece()
                moved = True
                self.game_state.actions_per_piece -= 1

        if moved and self.game_state.is_landing:
            # Reset lock timer if the piece moved or rotated while landing
            self.game_state.lock_timer = pygame.time.get_ticks()

    def update_game_state(self):
        current_time = pygame.time.get_ticks()
        time_delta = current_time - self.last_move_time

        if self.game_state.valid_position(adj_y=1):
            if time_delta > self.fall_speed * 1000:
                self.game_state.position[0] += 1
                self.last_move_time = current_time
                self.game_state.is_landing = False
                self.game_state.lock_timer = 0
        else:
            if not self.game_state.is_landing:
                self.game_state.is_landing = True
                self.game_state.lock_timer = current_time
            else:
                if current_time - self.game_state.lock_timer >= self.game_state.lock_delay:
                    self.game_state.lock_piece()
                    self.game_state.is_landing = False
                    self.game_state.lock_timer = 0

        if not self.render_mode or not self.manual:
            return

        self.render()
        self.clock.tick(60)

    # def calculate_reward(self) -> float:
    #     reward = 0
    #     # Good rewards
    #     reward += self.game_state.score * (self.weights.get('score') or .4)  # Scoring
    #     reward += self.game_state.lines_cleared * (self.weights.get('lines_cleared') or 300)  # Line clears
    #     for row in range(BOARD_HEIGHT):
    #         reward += self.reward_for_fill_level(row) * (self.weights.get('fill_level') or .4)  # Fill levels
    #     # Add more rewards for complex moves eventually

    #     # Bad rewards
    #     height = self.get_board_height()
    #     # Penalize height exponentially, but never if the height is under 3
    #     if height > 3:
    #         reward -= height ** 1.4 * (self.weights.get('height') or .4)
    #     holes = self.count_holes_dfs(self.game_state.board)
    #     reward -= holes * (self.weights.get('holes') or 50)  # Penalize holes
    #     # Penalize for using too many actions per piece
    #     if self.game_state.actions_per_piece > 5:
    #         reward -= self.game_state.actions_per_piece * (self.weights.get('actions_per_piece') or 1.5)
    #     # Penalize for bumpiness
    #     bumpiness = self.calculate_bumpiness() * (self.weights.get('actions_per_piece') or 1.5)
    #     reward -= bumpiness

    #     # Scale the absolute value of the reward to its logarithmic value to avoid excessing rewards or punishments
    #     reward = np.sign(reward) * np.log1p(abs(reward))

    #     return reward

    @torch.jit.script
    def calculate_reward(self) -> float:
        """Calculate reward using GPU operations
        
        Returns:
            float: Calculated reward value
        """
        reward = 0.0
        
        # Good rewards
        reward += self.game_state.score * (self.weights.get('score', 0.4))
        reward += self.game_state.lines_cleared * (self.weights.get('lines_cleared', 300.0))
        
        # Column heights
        heights = torch.argmax(self.game_state.board != 0, dim=0)
        max_height = BOARD_HEIGHT - torch.min(heights).item()
        
        # Penalize height if too high
        if max_height > 3:
            reward -= (max_height ** 1.4) * (self.weights.get('height', 0.4))
        
        # Holes penalty
        holes = self.count_holes()
        reward -= holes * (self.weights.get('holes', 50.0))
        
        # Bumpiness penalty 
        bumpiness = float(torch.sum(torch.abs(heights[:-1] - heights[1:])).item())
        reward -= bumpiness * (self.weights.get('bumpiness', 1.5))
        
        # Action penalty
        if self.game_state.actions_per_piece > 5:
            reward -= self.game_state.actions_per_piece * (self.weights.get('actions_per_piece', 1.5))
        
        # Log scale to avoid extreme values
        return float(torch.sign(torch.tensor(reward)) * torch.log1p(torch.abs(torch.tensor(reward))))

    @lru_cache(maxsize=1)
    def get_board_height(self) -> int:
        """Get current board height with caching
        
        Returns:
            int: Height of highest block
        """
        heights = torch.argmax(self.board != 0, dim=0)
        return int(BOARD_HEIGHT - torch.min(heights).item())

    def count_holes_dfs(self, board: np.ndarray) -> int:
        BOARD_WIDTH, BOARD_HEIGHT = board.shape[1], board.shape[0]
        visited = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=bool)
        stack = []
        
        # Determine the starting row based on pre-calculated board height
        height = self.get_board_height()
        start_row = max(0, BOARD_HEIGHT - height - 1)
        
        # Initialize stack with all empty cells in the starting row
        for x in range(BOARD_WIDTH):
            if board[start_row][x] == 0 and not visited[start_row][x]:
                stack.append((x, start_row))
                visited[start_row][x] = True
        
        # Iterative DFS
        while stack:
            x, y = stack.pop()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < BOARD_WIDTH and 0 <= ny < BOARD_HEIGHT:
                    if board[ny][nx] == 0 and not visited[ny][nx]:
                        stack.append((nx, ny))
                        visited[ny][nx] = True
        
        # Count holes: empty cells not reachable from the starting row
        holes = np.sum((board == 0) & (~visited))
        return int(holes)

    @torch.jit.script
    def count_holes(self):
        """Count holes using GPU operations
        
        Returns:
            int: Number of holes in board
        """
        # Find highest block in each column
        heights = torch.argmax(self.board != 0, dim=0)
        
        # Create mask of positions that should be filled
        rows = torch.arange(BOARD_HEIGHT, device='cuda')[:, None]
        cols = torch.arange(BOARD_WIDTH, device='cuda')[None, :]
        should_be_filled = rows >= heights
        
        # Count empty cells below heights
        is_empty = (self.board == 0)
        holes = torch.sum(should_be_filled & is_empty).item()
        
        return int(holes)
    
    def get_fill_level(self, row: int) -> int:
        return sum(self.game_state.board[row] > 0)
    
    def reward_for_fill_level(self, row: int) -> int:
        fill_level = self.get_fill_level(row)
        if fill_level > 3:
            return fill_level ** 2
        return 0
    
    def calculate_bumpiness(self):
        heights = [BOARD_HEIGHT - np.argmax(self.game_state.board[:, x]) if np.any(self.game_state.board[:, x]) else 0 for x in range(BOARD_WIDTH)]
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        return bumpiness

    # def get_state(self) -> np.ndarray:
    #     # Render the board to a surface
    #     board_surface = Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
    #     board_surface.fill(BLACK)

    #     # Draw game board in center, slightly above the bottom, with grid pattern
    #     for y in range(BOARD_HEIGHT):
    #         for x in range(BOARD_WIDTH):
    #             rect = pygame.Rect(self.board_x + x * BLOCK_SIZE, self.board_y + y * BLOCK_SIZE , BLOCK_SIZE, BLOCK_SIZE)
    #             pygame.draw.rect(board_surface, (40, 40, 40), rect, 1)
    #             if self.game_state.board[y][x]:
    #                 pygame.draw.rect(board_surface, COLORS[self.game_state.board[y][x]], rect)
    #                 pygame.draw.rect(board_surface, SECONDARY_COLORS[self.game_state.board[y][x]], 
    #                                     (self.board_x + x * BLOCK_SIZE + 2, self.board_y + y * BLOCK_SIZE + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))
                    
    #     # Draw current piece
    #     shape = self.game_state.current_piece['shape'][self.game_state.rotation_index]
    #     border_color = COLORS[PIECE_IDS[self.game_state.current_piece['type']]]
    #     color = SECONDARY_COLORS[PIECE_IDS[self.game_state.current_piece['type']]]
    #     for y, row in enumerate(shape):
    #         for x, cell in enumerate(row):
    #             if cell:
    #                 px = self.board_x + (self.game_state.position[1] + x) * BLOCK_SIZE
    #                 py = self.board_y + (self.game_state.position[0] + y) * BLOCK_SIZE
    #                 if py >= self.board_y:
    #                     border = pygame.Rect(px, py, BLOCK_SIZE, BLOCK_SIZE)
    #                     pygame.draw.rect(board_surface, border_color, border)

    #                     rect = pygame.Rect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
    #                     pygame.draw.rect(board_surface, color, rect)

    #     # Draw side panel background
    #     side_panel_rect = pygame.Rect(
    #         self.side_panel_x_left - self.side_panel_width_left,
    #         self.board_y + 35,
    #         self.side_panel_width_left,
    #         140
    #     )

    #     pygame.draw.rect(board_surface, (30, 30, 30), side_panel_rect)

    #     # Draw "Hold" label
    #     self.draw_text("Held:", (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 40), board_surface)

    #     # Draw held piece
    #     if self.game_state.hold_piece:
    #         self.draw_small_piece(self.game_state.hold_piece, (self.side_panel_x_left - self.side_panel_width_left + 50, self.board_y + 110), board_surface)

    #     # Draw a little box around the hold piece
    #     pygame.draw.rect(board_surface, WHITE, (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 80, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE), 1)

    #     # Draw "Next" label
    #     self.draw_text("Next", (self.side_panel_x_right + 20, self.board_y + 5), board_surface)

    #     # Draw next five pieces
    #     for i, piece in enumerate(self.game_state.next_pieces[:5]):
    #         self.draw_small_piece(piece, (self.side_panel_x_right + 40, self.board_y + 65 + i * 60), board_surface)

    #     # Convert the surface to a numpy array
    #     raw_str = pygame.image.tostring(board_surface, 'RGB')
    #     image = np.frombuffer(raw_str, dtype=np.uint8)
    #     image = image.reshape((SCREEN_HEIGHT, SCREEN_WIDTH, 3))
    #     if self.render_queue:
    #         self.render_queue.put(image)

    #     image = np.mean(image, axis=2)  # Convert to grayscale
    #     image = image.astype(np.uint8)

    #     return image
    def get_state(self) -> np.ndarray:
        """Get current state tensor optimized for GPU
        
        Returns:
            torch.Tensor: State representation (BOARD_HEIGHT, BOARD_WIDTH)
        """
        # Start with board state
        state = self.game_state.board.clone()
        
        # Add current piece
        piece = torch.tensor(self.game_state.current_piece['shape'][self.game_state.rotation_index],
                            device='cuda', dtype=torch.int8)
        y, x = self.game_state.position
        if y >= 0:  # Only draw piece if on board
            h, w = piece.shape
            state[y:y+h, x:x+w] |= piece
            
        return state

    def render(self):
        with self.lock:
            self.screen.fill(BLACK)

            # Draw game board in center, slightly above the bottom, with grid pattern
            for y in range(BOARD_HEIGHT):
                for x in range(BOARD_WIDTH):
                    rect = pygame.Rect(self.board_x + x * BLOCK_SIZE, self.board_y + y * BLOCK_SIZE , BLOCK_SIZE, BLOCK_SIZE)
                    pygame.draw.rect(self.screen, (40, 40, 40), rect, 1)
                    if self.game_state.board[y][x]:
                        pygame.draw.rect(self.screen, COLORS[self.game_state.board[y][x]], rect)
                        pygame.draw.rect(self.screen, SECONDARY_COLORS[self.game_state.board[y][x]], 
                                            (self.board_x + x * BLOCK_SIZE + 2, self.board_y + y * BLOCK_SIZE + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4))

            # Draw current piece
            shape = self.game_state.current_piece['shape'][self.game_state.rotation_index]
            border_color = COLORS[PIECE_IDS[self.game_state.current_piece['type']]]
            color = SECONDARY_COLORS[PIECE_IDS[self.game_state.current_piece['type']]]
            for y, row in enumerate(shape):
                for x, cell in enumerate(row):
                    if cell:
                        px = self.board_x + (self.game_state.position[1] + x) * BLOCK_SIZE
                        py = self.board_y + (self.game_state.position[0] + y) * BLOCK_SIZE
                        if py >= self.board_y:
                            border = pygame.Rect(px, py, BLOCK_SIZE, BLOCK_SIZE)
                            pygame.draw.rect(self.screen, border_color, border)

                            rect = pygame.Rect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
                            pygame.draw.rect(self.screen, color, rect)

            # Draw side panel background
            side_panel_rect = pygame.Rect(
                self.side_panel_x_left - self.side_panel_width_left,
                self.board_y + 35,
                self.side_panel_width_left,
                140
            )
            
            pygame.draw.rect(self.screen, (30, 30, 30), side_panel_rect)

            # Draw "Score" label
            self.draw_text(f"Score: {self.game_state.score}", (self.side_panel_x_left - self.side_panel_width_left, self.board_y + 5), self.screen)

            # Draw "Hold" label
            self.draw_text("Held:", (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 40), self.screen)

            # Draw held piece
            if self.game_state.hold_piece:
                self.draw_small_piece(self.game_state.hold_piece, (self.side_panel_x_left - self.side_panel_width_left + 50, self.board_y + 110), self.screen)

            # Draw a little box around the hold piece
            pygame.draw.rect(self.screen, WHITE, (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 80, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE), 1)

            # Draw "Next" label
            self.draw_text("Next", (self.side_panel_x_right + 20, self.board_y + 5), self.screen)

            # Draw next five pieces
            for i, piece in enumerate(self.game_state.next_pieces[:5]):
                self.draw_small_piece(piece, (self.side_panel_x_right + 40, self.board_y + 65 + i * 60), self.screen)

            pygame.display.flip()

    def draw_small_piece(self, piece, center, surface):
        type = piece['type']
        color = SECONDARY_COLORS[PIECE_IDS[type]]
        border_color = COLORS[PIECE_IDS[type]]
        small_block_size = BLOCK_SIZE // 1.5
        small_block_border = BLOCK_SIZE // 1.6
        match type:
            case 'I':
                # Draw horizontal, 4 cells
                for i in range(4):
                    y_offset = small_block_border // 2
                    x_start = center[0] - small_block_border * 2
                    for j in range(4):
                        x = x_start + j * small_block_border
                        y = center[1] - y_offset
                        pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                        pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))
            case 'O':
                # Draw 2x2 square
                for i in range(2):
                    for j in range(2):
                        x = center[0] - small_block_border + j * small_block_border
                        y = center[1] - small_block_border + i * small_block_border
                        pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                        pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))
            case 'T':
                # Draw T shape
                for i in range(3):
                    for j in range(3):
                        if (i == 0 and j == 1) or (i == 1 and j == 0) or (i == 1 and j == 1) or (i == 1 and j == 2):
                            x = center[0] - small_block_border + j * small_block_border
                            y = center[1] - small_block_border + i * small_block_border
                            pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                            pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))
            case 'S':
                # Draw S shape
                for i in range(3):
                    for j in range(3):
                        if (i == 0 and j == 1) or (i == 1 and j == 0) or (i == 1 and j == 1) or (i == 2 and j == 0):
                            x = center[0] - small_block_border + j * small_block_border
                            y = center[1] - small_block_border + i * small_block_border
                            pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                            pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))
            case 'Z':
                # Draw Z shape
                for i in range(3):
                    for j in range(3):
                        if (i == 0 and j == 0) or (i == 1 and j == 0) or (i == 1 and j == 1) or (i == 2 and j == 1):
                            x = center[0] - small_block_border + j * small_block_border
                            y = center[1] - small_block_border + i * small_block_border
                            pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                            pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))
            case 'J':
                # Draw J shape
                for i in range(3):
                    for j in range(3):
                        if (i == 0 and j == 1) or (i == 1 and j == 1) or (i == 2 and j == 0) or (i == 2 and j == 1):
                            x = center[0] - small_block_border + j * small_block_border
                            y = center[1] - small_block_border + i * small_block_border
                            pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                            pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))
            case 'L':
                # Draw L shape
                for i in range(3):
                    for j in range(3):
                        if (i == 0 and j == 0) or (i == 1 and j == 0) or (i == 2 and j == 0) or (i == 2 and j == 1):
                            x = center[0] - small_block_border + j * small_block_border
                            y = center[1] - small_block_border + i * small_block_border
                            pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                            pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))



    def draw_text(self, text, position, surface):
        font = pygame.font.SysFont('Arial', 24)
        text_surface = font.render(text, True, WHITE)
        surface.blit(text_surface, position)

    def close(self):
        pygame.quit()