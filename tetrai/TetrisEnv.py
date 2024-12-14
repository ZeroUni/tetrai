import pygame
import random
import numpy as np
from pygame import Surface
from typing import Tuple, Dict

import threading
import multiprocessing

import torch
import cupy as cp
from numba import jit, cuda
from functools import lru_cache
import gc

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

# Define piece sets for each level
LEVEL_PIECES = {
    1: ['O', 'I'],
    2: ['O', 'I', 'L', 'J', 'T'],
    3: ['O', 'I', 'L', 'J', 'T', 'S', 'Z']
}

class GameState:
    def __init__(self, level=1):
        # Pre-allocate tensors and pin them to GPU
        self.board = torch.zeros((BOARD_HEIGHT, BOARD_WIDTH), 
                               dtype=torch.int8,
                               device='cuda')
        self.temp_board = torch.zeros_like(self.board)
        self.visited = torch.zeros_like(self.board, dtype=torch.bool)
        
        # Cache tensors for piece operations
        self.piece_matrix = torch.zeros((4, 4), dtype=torch.int8, device='cuda')
        self.position_tensor = torch.zeros(2, dtype=torch.int32, device='cuda')
        
        # Cache piece matrices as tensors
        self.piece_tensors = {
            piece_type: torch.tensor(shapes, device='cuda', dtype=torch.int8)
            for piece_type, shapes in SHAPES.items()
        }
        
        self.level = level
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
        self.give_reward = False

    def get_new_piece(self):
        available_pieces = LEVEL_PIECES[self.level]
        shape_type = random.choice(available_pieces)
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

    def valid_position(self, piece=None, adj_x=0, adj_y=0):
        """GPU-optimized position validation"""
        if piece is None:
            piece = self.current_piece['shape'][self.rotation_index]
            piece = torch.tensor(piece, device='cuda', dtype=torch.int8)
        elif isinstance(piece, list):
            piece = torch.tensor(piece, device='cuda', dtype=torch.int8)
                
        # Use cached piece matrix
        self.piece_matrix.zero_()
        self.piece_matrix[:piece.size(0), :piece.size(1)].copy_(piece)
        
        # Calculate new position using cached tensor
        self.position_tensor[0] = self.position[0] + adj_y
        self.position_tensor[1] = self.position[1] + adj_x
        
        # Get piece bounds efficiently
        non_zero_y, non_zero_x = torch.where(self.piece_matrix == 1)
        
        # Calculate board positions
        board_y = self.position_tensor[0] + non_zero_y
        board_x = self.position_tensor[1] + non_zero_x
        
        # Check bounds in single operation
        if (torch.any(board_x >= BOARD_WIDTH) or
            torch.any(board_x < 0) or
            torch.any(board_y >= BOARD_HEIGHT)):
            return False
                
        # Filter valid board positions
        valid_mask = board_y >= 0
        board_y = board_y[valid_mask]
        board_x = board_x[valid_mask]
        
        if len(board_y) == 0:
            return True
                
        return not torch.any(self.board[board_y, board_x] != 0)

    def lock_piece(self):
        self.can_hold = False
        self.give_reward = True
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
        
        # Score calculation using if/elif instead of match/case
        if num_cleared == 1:
            self.score += 40
        elif num_cleared == 2:
            self.score += 500
        elif num_cleared == 3:
            self.score += 1200
        elif num_cleared == 4:
            self.score += 2800
        
        # Level up bonus
        if self.lines_cleared >= 10 * (lines_before // 10 + 1):
            self.score += 1000
            
        return num_cleared

class TetrisEnv:
    def __init__(self, display_manager=None, max_moves=-1, weights=None, manual=False, level=1):
        try:
            if not pygame.get_init():
                pygame.init()
        except pygame.error as e:
            print(f"Failed to initialize PyGame: {e}")
            raise

        # Use simpler state processor architecture
        self.state_processor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 1, 3, padding=1, bias=False),  # Remove bias for efficiency
            torch.nn.ReLU()
        ).cuda()
        torch.nn.init.ones_(self.state_processor[0].weight)  # Initialize with ones for stable processing

        # Pre-allocate CUDA tensors for state processing
        self.state_tensor = torch.zeros((1, 1, BOARD_HEIGHT, BOARD_WIDTH), device='cuda')
        
        # Pre-allocate surfaces for rendering
        self.board_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.piece_surface = pygame.Surface((BLOCK_SIZE * 4, BLOCK_SIZE * 4))

        if not pygame.get_init():
            pygame.init()
        
        self.manual = manual
        if self.manual:
            self.screen = pygame.display.set_mode((int(SCREEN_WIDTH), int(SCREEN_HEIGHT)))
            pygame.display.set_caption('TetrisEnv')
        else:
            self.screen = Surface((SCREEN_WIDTH, SCREEN_HEIGHT))

        self.clock = pygame.time.Clock()
        self.level = min(max(1, level), 3)  # Ensure level is between 1 and 3
        self.game_state = GameState(level=self.level)
        self.fall_time = 0
        self.fall_speed = 0.5  # Seconds per fall
        self.last_move_time = pygame.time.get_ticks()
        self.render_mode = False  # Set to False to run without rendering
        self.render_delay = 100  # Delay between renders in milliseconds
        self.last_render_time = pygame.time.get_ticks()

        self.lock = threading.Lock()
        self.display_manager = display_manager

        self.max_moves = max_moves
        self.move_count = 0

        self.weights = weights or {}
        print(f"Using weights: {self.weights}")
        weights_list = [
            self.weights.get('score', 1.0),
            self.weights.get('lines_cleared', 10.0), 
            self.weights.get('fill_level', 1.0),
            self.weights.get('height', 1.0),
            self.weights.get('holes', 1.0),
            self.weights.get('bumpiness', 1.0),
            self.weights.get('actions_per_piece', 1.0),
            self.weights.get('game_over', 1.0)
        ]
        self.tensor_weights = torch.tensor(weights_list, device='cuda', dtype=torch.float32)

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

        # Pre render the static elements of the baord
        self.static_background = self.render_static_background()

    def render_static_background(self) -> Surface:
        static_background = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        static_background.fill(BLACK)
        
        # Draw static grid lines
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                rect = (self.board_x + x * BLOCK_SIZE,
                    self.board_y + y * BLOCK_SIZE,
                    BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(static_background, (40, 40, 40), rect, 1)
        
        # Draw static side panels
        side_panel_rect = pygame.Rect(
            self.side_panel_x_left - self.side_panel_width_left,
            self.board_y + 35,
            self.side_panel_width_left,
            140
        )
        pygame.draw.rect(static_background, (30, 30, 30), side_panel_rect)
        
        # Pre-render static text
        font = pygame.font.SysFont('Arial', 24)
        held_text = font.render("Held:", True, WHITE)
        next_text = font.render("Next", True, WHITE)
        static_background.blit(held_text, 
            (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 40))
        static_background.blit(next_text, 
            (self.side_panel_x_right + 20, self.board_y + 5))
        
        # Draw hold piece box
        pygame.draw.rect(static_background, WHITE,
                        (self.side_panel_x_left - self.side_panel_width_left + 10,
                        self.board_y + 80, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE), 1)
        
        return static_background

    def reset(self) -> np.ndarray:
        with self.lock:
            # Clear CUDA memory
            if hasattr(self, 'state_tensor'):
                self.state_tensor.zero_()
            self.game_state = GameState(level=self.level)
            self.fall_time = 0
            self.last_move_time = pygame.time.get_ticks()
            self.move_count = 0
            return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        with self.lock:
            reward = 0
            done = False

            # Apply action
            actions_per_piece = self.game_state.actions_per_piece
            self.apply_action(action)

            # Update game state
            self.update_game_state()

            # Calculate reward
            if self.game_state.give_reward:
                reward += calculate_reward(self.game_state.board, self.game_state.score, self.game_state.lines_cleared, self.tensor_weights, BOARD_HEIGHT, BOARD_WIDTH, actions_per_piece)
                self.game_state.give_reward = False

            if self.game_state.game_over:
                done = True
                reward -= self.weights.get('game_over') or 1000  # Penalty for dying

            self.move_count += 1
            if 0 < self.max_moves <= self.move_count:
                reward -= pow(actions_per_piece, 3) * 0.8  # Penalty for taking too long
                done = True  # End episode due to max moves

            return self.get_state(), reward, done

    def apply_action(self, action: int):
        moved = False
        self.game_state.actions_per_piece += 1
        match action:
            case 0:  # Hard Drop
                while self.game_state.valid_position(adj_y=1):
                    self.game_state.position[0] += 1
                self.game_state.lock_piece()
                self.game_state.is_landing = False
                self.game_state.lock_timer = 0
                self.game_state.actions_per_piece -= 1
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
            case 6:  # Hold
                self.game_state.hold_current_piece()
                moved = True
                self.game_state.actions_per_piece -= 1
            case 7:  # Nothing
                return

        if moved and self.game_state.is_landing:
            # Reset lock timer if the piece moved or rotated while landing
            self.game_state.lock_timer = pygame.time.get_ticks()

    def get_legal_actions(self) -> list:
        """Get list of legal actions in current state
        
        Returns:
            list: List of legal action indices (0-7)
        """
        legal = []
        
        # Hard drop is always legal
        legal.append(0)
        
        # Check rotations
        # Right rotation
        temp_rot = (self.game_state.rotation_index + 1) % 4
        if self.check_rotation(temp_rot):
            legal.append(1)
            
        # Left rotation  
        temp_rot = (self.game_state.rotation_index - 1) % 4
        if self.check_rotation(temp_rot):
            legal.append(2)
        
        # Down move
        if self.game_state.valid_position(adj_y=1):
            legal.append(3)
            
        # Left move
        if self.game_state.valid_position(adj_x=-1):
            legal.append(4)
            
        # Right move
        if self.game_state.valid_position(adj_x=1):
            legal.append(5)
            
        # Do nothing is always legal
        legal.append(6)
        
        # Hold is legal if not already used
        if self.game_state.can_hold:
            legal.append(7)
            
        return legal
    
    def get_legal_actions_mask(self) -> torch.Tensor:
        """Get binary mask of legal actions
        
        Returns:
            torch.Tensor: Binary mask where 1 indicates legal action
        """
        # Initialize mask with zeros
        mask = torch.zeros(8, dtype=torch.float16, device='cuda')
        
        # Hard drop and do nothing always legal
        mask[0] = 1  # Hard drop
        mask[6] = 1  # Do nothing
        
        # Check rotations
        # Right rotation
        temp_rot = (self.game_state.rotation_index + 1) % 4
        if self.check_rotation(temp_rot):
            mask[1] = 1
            
        # Left rotation
        temp_rot = (self.game_state.rotation_index - 1) % 4  
        if self.check_rotation(temp_rot):
            mask[2] = 1
        
        # Down move
        if self.game_state.valid_position(adj_y=1):
            mask[3] = 1
            
        # Left move
        if self.game_state.valid_position(adj_x=-1):
            mask[4] = 1
            
        # Right move
        if self.game_state.valid_position(adj_x=1):
            mask[5] = 1
            
        # Hold piece
        if self.game_state.can_hold:
            mask[7] = 1
            
        return mask
    
    def check_rotation(self, new_rotation: int) -> bool:
        """Helper to check if rotation is valid considering wall kicks
        
        Args:
            new_rotation: Target rotation index
            
        Returns:
            bool: True if rotation is valid
        """
        old_rotation = self.game_state.rotation_index
        old_position = self.game_state.position.copy()
        rotated_shape = self.game_state.current_piece['shape'][new_rotation]
    
        # Get wall kicks for this rotation transition
        rotation_transition = (old_rotation, new_rotation) 
        kicks = WALL_KICKS[self.game_state.current_piece['type']].get(rotation_transition, [])
    
        # Try each wall kick
        for dx, dy in kicks:
            test_x = old_position[1] + dx
            test_y = old_position[0] + dy
            
            # Create temp position and check
            test_pos = [test_y, test_x]
            if self.game_state.valid_position(piece=rotated_shape, 
                                            adj_x=test_x-old_position[1],
                                            adj_y=test_y-old_position[0]):
                return True
                
        return False

    def update_game_state(self):
        current_time = pygame.time.get_ticks()
        time_delta = current_time - self.last_move_time

        # Check if piece can move down
        if self.game_state.valid_position(adj_y=1):
            # Only move down if enough time has passed
            if time_delta > self.fall_speed * 1000:
                self.game_state.position[0] += 1
                self.last_move_time = current_time
                self.game_state.is_landing = False
                self.game_state.lock_timer = 0
        else:
            # Piece can't move down - handle landing
            if not self.game_state.is_landing:
                # Start landing sequence
                self.game_state.is_landing = True
                self.game_state.lock_timer = current_time
            elif current_time - self.game_state.lock_timer >= self.game_state.lock_delay:
                # Lock delay expired - lock the piece
                if not self.game_state.valid_position(adj_y=1):  # Double check we can't move down
                    self.game_state.lock_piece()
                    self.last_move_time = current_time  # Reset fall timer
                else:
                    # If we can suddenly move down again, cancel landing
                    self.game_state.is_landing = False
                    self.game_state.lock_timer = 0

        if self.render_mode and self.manual:
            self.render()
            self.clock.tick(60)

    @lru_cache(maxsize=1)
    def get_board_height(self) -> int:
        """Get current board height with caching
        
        Returns:
            int: Height of highest block
        """
        heights = torch.argmax(self.board != 0, dim=0)
        return int(BOARD_HEIGHT - torch.min(heights).item())

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

    def get_state(self) -> np.ndarray:
        """Get game state as visual representation optimized for GPU
        
        Returns:
            np.ndarray: Visual state representation (HEIGHT, WIDTH, 3)
        """
        # Use pre-allocated surface
        self.screen.blit(self.static_background, (0, 0))
        
        # Convert board state to CPU for drawing
        board_array = self.game_state.board.cpu().numpy()
        
        # Batch draw board cells
        board_positions = []
        board_colors = []
        board_secondary_colors = []
        
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if board_array[y][x]:
                    rect = (self.board_x + x * BLOCK_SIZE, 
                        self.board_y + y * BLOCK_SIZE,
                        BLOCK_SIZE, BLOCK_SIZE)
                    board_positions.append(rect)
                    board_colors.append(COLORS[board_array[y][x]])
                    board_secondary_colors.append(SECONDARY_COLORS[board_array[y][x]])
                
        # Draw filled cells in batches
        for rect, color, sec_color in zip(board_positions, board_colors, board_secondary_colors):
            pygame.draw.rect(self.screen, color, rect)
            inner_rect = (rect[0] + 2, rect[1] + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
            pygame.draw.rect(self.screen, sec_color, inner_rect)
        
        # Draw current piece efficiently
        shape = self.game_state.current_piece['shape'][self.game_state.rotation_index]
        piece_id = PIECE_IDS[self.game_state.current_piece['type']]
        border_color = COLORS[piece_id]
        color = SECONDARY_COLORS[piece_id]
        
        # Batch piece drawing
        piece_positions = []
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    px = self.board_x + (self.game_state.position[1] + x) * BLOCK_SIZE
                    py = self.board_y + (self.game_state.position[0] + y) * BLOCK_SIZE
                    if py >= self.board_y:
                        piece_positions.append((px, py))
        
        # Draw piece borders and fills in batches
        for px, py in piece_positions:
            border = pygame.Rect(px, py, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, border_color, border)
            rect = pygame.Rect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
            pygame.draw.rect(self.screen, color, rect)
        
        if self.game_state.hold_piece:
            self.draw_small_piece(
                self.game_state.hold_piece,
                (self.side_panel_x_left - self.side_panel_width_left + 50, self.board_y + 110),
                self.screen
            )
                
        # Draw next pieces efficiently
        for i, piece in enumerate(self.game_state.next_pieces[:5]):
            self.draw_small_piece(
                piece,
                (self.side_panel_x_right + 40, self.board_y + 65 + i * 60),
                self.screen
            )
        
        # Convert surface to numpy array efficiently
        image = pygame.surfarray.array3d(self.screen)
        image = np.flip(image, axis=0)  # Flip vertically 
        image = np.rot90(image, -1, axes=(0, 1))

        if self.display_manager:
            self.display_manager.update(image.copy())
        
        # Convert to grayscale using optimized numpy operations
        image = np.mean(image, axis=2, dtype=np.uint8)

        return image

    def render(self):
        with self.lock:
            if not self.manual:
                return

            # Use our state func to get the image to draw
            state = self.get_state()
            # Display the pygame screen
            pygame.display.flip()

    def draw_small_piece(self, piece, center, surface):
        """Simplified piece rendering with pre-calculated patterns"""
        type = piece['type']
        pattern = SHAPES[type][0]  # Use first rotation state
        color = SECONDARY_COLORS[PIECE_IDS[type]]
        border_color = COLORS[PIECE_IDS[type]]
        small_block_size = BLOCK_SIZE // 1.5
        small_block_border = BLOCK_SIZE // 1.6

        for i, row in enumerate(pattern):
            for j, cell in enumerate(row):
                if cell:
                    x = center[0] - small_block_border + j * small_block_border
                    y = center[1] - small_block_border + i * small_block_border
                    pygame.draw.rect(surface, border_color, (x, y, small_block_border, small_block_border))
                    pygame.draw.rect(surface, color, (x + 1, y + 1, small_block_size, small_block_size))

    def draw_text(self, text, position, surface):
        font = pygame.font.SysFont('Arial', 24)
        text_surface = font.render(text, True, WHITE)
        surface.blit(text_surface, position)

    def get_score(self) -> int:
        return self.game_state.score

    def close(self):
        # Proper cleanup of CUDA resources
        if hasattr(self, 'state_processor'):
            self.state_processor.cpu()
            del self.state_processor
        if hasattr(self, 'state_tensor'):
            del self.state_tensor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        pygame.quit()
        gc.collect()

    def increase_level(self):
        """Increase curriculum level if not at max"""
        if self.level < 3:
            self.level += 1
            # Update game state with new level
            self.game_state.level = self.level
            return True
        return False

    def get_level(self) -> int:
        """Get current curriculum level"""
        return self.level

@torch.jit.script
def count_holes(board, heights, BOARD_HEIGHT: int, BOARD_WIDTH: int) -> int:
    """Count holes using DFS on GPU tensors
    
    Returns:
        int: Number of holes in board
    """
    visited = torch.zeros_like(board, dtype=torch.bool, device='cuda')
    
    # Start from top row of each column
    for col in range(BOARD_WIDTH):
        height = int(heights[col].item()) # Convert to int explicitly
        row = max(0, BOARD_HEIGHT - height) # Use max() to avoid negative values
        
        if row < BOARD_HEIGHT and board[row, col] == 0:
            # Flood fill from this empty cell
            stack = [(row, col)]
            while stack:
                r, c = stack.pop()
                
                # Skip if out of bounds or already visited
                if (r < 0 or r >= BOARD_HEIGHT or 
                    c < 0 or c >= BOARD_WIDTH or
                    visited[r, c]):
                    continue
                    
                visited[r, c] = True
                
                # Only spread to empty cells
                if board[r, c] == 0:
                    # Add adjacent cells to stack
                    for nr, nc in [(r+1, c), (r-1, c), (r, c+1), (r, c-1)]:
                        if (0 <= nr < BOARD_HEIGHT and 
                            0 <= nc < BOARD_WIDTH and 
                            not visited[nr, nc]):
                            stack.append((nr, nc))
    
    # Count holes - empty unvisited cells below highest block
    holes = 0
    for col in range(BOARD_WIDTH):
        height = int(heights[col].item())
        if height > 0:
            start_row = BOARD_HEIGHT - height
            holes += int(torch.sum((board[start_row:, col] == 0) & 
                                 (~visited[start_row:, col])).item())
                    
    return holes

@torch.jit.script
def calculate_heights(board, BOARD_HEIGHT: int) -> torch.Tensor:
    """Calculate heights of each column
    
    Args:
        board: Board tensor
    
    Returns:
        torch.Tensor: Height of each column
    """
    heights = torch.zeros(board.shape[1], dtype=torch.int32, device='cuda')
    
    for col in range(board.shape[1]):
        # Find first non-zero element from top
        col_data = board[:, col]
        non_zero = torch.nonzero(col_data)
        if len(non_zero) > 0:
            # Get height from top (first non-zero position)
            first_block = non_zero[0].item()
            heights[col] = BOARD_HEIGHT - first_block
            
    return heights

@torch.jit.script
def calc_fill_reward(board, BOARD_HEIGHT: int, BOARD_WIDTH: int) -> float:
    """Calculate reward for filling rows"""
    fill_reward = 0.
    for row in range(BOARD_HEIGHT):
        fill_reward += int(torch.sum(board[row] > 0).item()) ** 2
    return fill_reward

@torch.jit.script
def calculate_reward(board, score: int, lines: int, weights, BOARD_HEIGHT: int, BOARD_WIDTH: int, actions_taken: int) -> float:
    """Calculate reward using GPU operations"""
    heights = calculate_heights(board, BOARD_HEIGHT)
    holes = torch.tensor(float(count_holes(board, heights, BOARD_HEIGHT, BOARD_WIDTH)), device='cuda')
    bumpiness = torch.tensor(float(torch.sum(torch.abs(heights[:-1] - heights[1:])).item()), device='cuda')
    fill_reward = torch.tensor(float(calc_fill_reward(board, BOARD_HEIGHT, BOARD_WIDTH)), device='cuda')
    max_height = torch.tensor(float(torch.max(heights).item()), device='cuda')
    # If the value of max_height is less than 4, set it to 0 to avoid penalizing low height
    max_height = torch.where(max_height < 5, torch.tensor(0, device='cuda'), max_height)
    
    # Convert score and lines to tensors
    score_t = torch.tensor(float(score), device='cuda')
    lines_t = torch.tensor(float(lines), device='cuda')
    
    # Calculate individual components
    score_component = score_t * weights[0]
    lines_component = lines_t * weights[1]
    fill_component = fill_reward * weights[2]
    height_component = torch.pow(max_height, 3) * weights[3]
    holes_component = holes * weights[4]
    bumpiness_component = bumpiness * weights[5]
    if actions_taken > 8:
        actions_component = (actions_taken * actions_taken) * weights[6]
    else:
        actions_component = torch.tensor(0., device='cuda')

    # Print out the components for debugging
    # print(f"Score: {score_component.item()} Lines: {lines_component.item()} Fill: {fill_component.item()} Height: {height_component.item()} Holes: {holes_component.item()} Bumpiness: {bumpiness_component.item()} Actions: {actions_component.item()}")
    
    # Sum components
    reward = score_component + lines_component + fill_component - height_component - holes_component - bumpiness_component - actions_component    
    return float(reward)