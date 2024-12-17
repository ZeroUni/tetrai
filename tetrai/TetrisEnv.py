import pygame
import random
import numpy as np
from pygame import Surface
from typing import Tuple, Dict, List, Optional

import threading
import multiprocessing

import torch
import torchvision.transforms as T
import torchvision.utils as vutils
import torch.nn.functional as F
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

# Correct piece IDs to match standard Tetris colors
PIECE_IDS = {
    'I': 1,  # Cyan (light blue)
    'O': 2,  # Yellow
    'T': 3,  # Purple
    'S': 4,  # Green
    'Z': 5,  # Red
    'J': 6,  # Blue
    'L': 7,  # Orange
}

# Define colors for each shape (maintaining standard Tetris colors)
COLORS = {
    1: (0, 255, 255),    # Cyan (I)
    2: (255, 255, 0),    # Yellow (O)
    3: (128, 0, 128),    # Purple (T)
    4: (0, 255, 0),      # Green (S)
    5: (255, 0, 0),      # Red (Z)
    6: (0, 0, 255),      # Blue (J)
    7: (255, 165, 0),    # Orange (L)
}

SECONDARY_COLORS = {
    1: (0, 200, 200),    # Cyan (I)
    2: (200, 200, 0),    # Yellow (O)
    3: (100, 0, 100),    # Purple (T)
    4: (0, 200, 0),      # Green (S)
    5: (200, 0, 0),      # Red (Z)
    6: (0, 0, 200),      # Blue (J)
    7: (200, 100, 0),    # Orange (L)
}

# Update SHAPES to match PIECE_IDS order
SHAPES = {
    'I': [   # ID: 1 - Cyan
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
    'O': [   # ID: 2 - Yellow
        [[1, 1],
         [1, 1]],
        [[1, 1],
         [1, 1]],
        [[1, 1],
         [1, 1]],
        [[1, 1],
         [1, 1]]
    ],
    'T': [   # ID: 3 - Purple
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
    'S': [   # ID: 4 - Green
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
    'Z': [   # ID: 5 - Red
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
    ],
    'J': [   # ID: 6 - Blue
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
    'L': [   # ID: 7 - Orange
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

@torch.jit.script
def validate_move(board: torch.Tensor, piece: torch.Tensor, position: List[int], 
                 adj_x: int = 0, adj_y: int = 0) -> bool:
    """GPU-optimized position validation
    
    Args:
        board: Game board tensor
        piece: Piece shape tensor
        position: [y, x] position
        adj_x: x adjustment
        adj_y: y adjustment
    
    Returns:
        bool: True if position is valid
    """
    # Pre-allocate piece matrix for consistency
    piece_matrix = torch.zeros((4, 4), dtype=torch.int8, device='cuda')
    piece_matrix[:piece.size(0), :piece.size(1)].copy_(piece)
    
    # Calculate new position
    new_y = position[0] + adj_y
    new_x = position[1] + adj_x
    
    # Get piece bounds efficiently
    non_zero_y, non_zero_x = torch.where(piece_matrix == 1)
    
    # Calculate board positions
    board_y = new_y + non_zero_y
    board_x = new_x + non_zero_x
    
    # Check bounds in single operation
    if (torch.any(board_x >= board.size(1)) or
        torch.any(board_x < 0) or
        torch.any(board_y >= board.size(0))):
        return False
            
    # Filter valid board positions
    valid_mask = board_y >= 0
    board_y = board_y[valid_mask]
    board_x = board_x[valid_mask]
    
    if len(board_y) == 0:
        return True
            
    return not torch.any(board[board_y, board_x] != 0)

@torch.jit.script
def calc_fill_reward(board) -> float:
    """Calculate reward for filling rows"""
    fill_reward = 0.
    for row in range(20):
        fill_reward += int(torch.sum(board[row] > 0).item()) ** 2
    return fill_reward

@torch.jit.script
def calc_fill_batch(boards: torch.Tensor) -> torch.Tensor:
    """Calculate fill rewards for multiple boards"""
    rewards = torch.zeros(boards.size(0), device='cuda')
    for i in range(boards.size(0)):
        rewards[i] = calc_fill_reward(boards[i])
    return rewards

@torch.jit.script
def evaluate_positions_batch(boards: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
    """Evaluate multiple board positions"""
    batch_size = boards.size(0)
    scores = torch.zeros(batch_size, device='cuda')
    
    # Calculate aggregate features
    max_heights = torch.max(heights, dim=1)[0].float()
    avg_heights = torch.mean(heights.float(), dim=1)
    bumpiness = torch.sum(torch.abs(heights[:, :-1] - heights[:, 1:]).float(), dim=1)
        
    # Count filled cells and complete lines
    filled = calc_fill_batch(boards)
    # For each board, set the number of complete lines to be equal to the number of rows with no 0s
    complete_lines = torch.sum(torch.all(boards != 0, dim=1).float(), dim=1)
    
    # Rebalanced weights with aggressive height control
    scores = (
        complete_lines * 500.0 +  
        filled * 10.0 - 
        torch.pow(max_heights, 2.0) * 50.0 -  
        avg_heights * 10.0 -  
        bumpiness * 5.0  
    )
        
    return scores

# Add new utility functions at the top level
@torch.jit.script
def batch_validate_moves(board: torch.Tensor, pieces: torch.Tensor, 
                        positions: torch.Tensor) -> torch.Tensor:
    """Validate multiple positions in parallel"""
    batch_size = positions.size(0)
    valid = torch.ones(batch_size, dtype=torch.bool, device='cuda')
    
    # Create expanded board for broadcasting
    board_exp = board.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Calculate piece positions without using as_tuple
    nonzero_indices = torch.nonzero(pieces)  # Returns [N, 3] tensor of indices
    piece_offsets = nonzero_indices[:, 1:]  # Keep only y,x coordinates
    
    # Expand positions for broadcasting with piece offsets
    expanded_positions = positions.unsqueeze(1).expand(-1, piece_offsets.size(0), -1)
    
    # Add offsets to get all piece positions
    piece_positions = expanded_positions + piece_offsets.unsqueeze(0)
    
    # Check bounds for all positions at once
    valid &= torch.all(
        (piece_positions[:, :, 0] >= 0) & 
        (piece_positions[:, :, 0] < board.size(0)) &
        (piece_positions[:, :, 1] >= 0) &
        (piece_positions[:, :, 1] < board.size(1)),
        dim=1
    )
    
    # Check collisions for valid positions
    valid_idx = torch.where(valid)[0]
    if len(valid_idx) > 0:
        valid_positions = piece_positions[valid_idx]
        board_values = board_exp[
            valid_idx.unsqueeze(1).expand(-1, valid_positions.size(1)),
            valid_positions[:, :, 0],
            valid_positions[:, :, 1]
        ]
        valid[valid_idx] = ~torch.any(board_values != 0, dim=1)
    
    return valid

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

        # Add new fields for target tracking
        self.target_x = None
        self.target_rot = None
        self.has_target = False

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
                
        return validate_move(self.board, piece, self.position, adj_x, adj_y)

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

        # Reset target position when piece locks
        self.target_x = None
        self.target_rot = None
        self.has_target = False

    def clear_lines(self):
        """Clear full lines using GPU operations
        
        Returns:
            int: Number of lines cleared
        """
        # Find full lines using GPU
        full_lines = torch.all(self.board != 0, dim=1)
        num_cleared = int(torch.sum(full_lines).item())
        
        if (num_cleared == 0):
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

class TorchRenderer:
    def __init__(self, width, height, block_size, device='cuda'):
        self.width = width
        self.height = height
        self.block_size = block_size
        self.device = device

        # Pre-compute side panel dimensions
        self.board_x = SCREEN_WIDTH // 2 - (BOARD_WIDTH * BLOCK_SIZE) // 2
        self.board_y = SCREEN_HEIGHT - (BOARD_HEIGHT * BLOCK_SIZE) - BOARD_MARGIN
        self.side_panel_left = self.board_x - 160  # Move left panel even further left
        self.side_panel_right = self.board_x + (BOARD_WIDTH * BLOCK_SIZE) + 20  # Right panel x position
        self.panel_color = torch.tensor([30, 30, 30], dtype=torch.uint8, device=device)

        # Add grid colors
        self.grid_colors = {
            'outer': torch.tensor([40, 40, 40], dtype=torch.uint8, device=device),
            'inner': torch.tensor([0, 0, 0], dtype=torch.uint8, device=device)
        }


        # Pre-compute static elements
        self.grid_template = self._create_grid_block_template()
        self.static_background = self._create_static_background()
        self.piece_templates = self._create_piece_templates()
        self.grid_template = self._create_grid_template()
        
        # Create reusable tensor for the game board
        self.board_tensor = torch.zeros((height, width, 3), 
                                      dtype=torch.uint8, 
                                      device=device)
                                            
        # Add piece offset lookup for centering
        self.piece_offsets = {
            'I': {'x': -0.5, 'y': 0.5},
            'O': {'x': -0.5, 'y': 0.0},   # O piece needs special handling
            'J': {'x': -0.5, 'y': 0.5},
            'L': {'x': -0.5, 'y': 0.5},
            'S': {'x': -0.5, 'y': 0.5},
            'T': {'x': -0.5, 'y': 0.5},
            'Z': {'x': -0.5, 'y': 0.5}
        }
        
        # Convert color dictionaries to tensors
        self.colors = {
            k: torch.tensor(v, dtype=torch.uint8, device=device)
            for k, v in COLORS.items()
        }
        self.secondary_colors = {
            k: torch.tensor(v, dtype=torch.uint8, device=device)
            for k, v in SECONDARY_COLORS.items()
        }
        
        # Pre-allocate output tensor
        self.output_tensor = torch.zeros(
            (SCREEN_HEIGHT, SCREEN_WIDTH, 3),
            dtype=torch.uint8,
            device=device
        )

                
    def _create_grid_block_template(self):
        """Create a template for a single grid block"""
        block = torch.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), 
                          dtype=torch.uint8,
                          device=self.device)
        
        # Fill outer region with grid color
        block[:, :] = self.grid_colors['outer']
        
        # Fill inner region with black
        inner_margin = 2
        block[inner_margin:-inner_margin, 
             inner_margin:-inner_margin] = self.grid_colors['inner']
            
        return block

    def _create_static_background(self):
        """Create static background with grid pattern"""
        background = torch.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), 
                               dtype=torch.uint8, 
                               device=self.device)
        
        # Calculate board region
        board_height = BOARD_HEIGHT * BLOCK_SIZE
        board_width = BOARD_WIDTH * BLOCK_SIZE
        
        # Draw grid blocks for the entire board area
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                px = self.board_x + x * BLOCK_SIZE
                py = self.board_y + y * BLOCK_SIZE
                
                if (px + BLOCK_SIZE <= SCREEN_WIDTH and py + BLOCK_SIZE <= SCREEN_HEIGHT):
                    background[py:py+BLOCK_SIZE, 
                             px:px+BLOCK_SIZE] = self.grid_template
        
        # Draw side panel backgrounds (dark gray)
        panel_width = 140
        y_start = self.board_y + 15
        y_end = self.board_y + 180
        
        # Left panel
        background[y_start:y_end, 20:self.board_x-20] = self.panel_color

        # Draw hold piece box border (white)
        box_size = 4 * BLOCK_SIZE
        hold_x = self.side_panel_left + 40
        hold_y = self.board_y + 70
        
        # Draw box borders with white color
        border_color = torch.tensor([255, 255, 255], dtype=torch.uint8, device=self.device)
        border_thickness = 2
        
        # Draw horizontal lines
        background[hold_y:hold_y+border_thickness, hold_x:hold_x+box_size] = border_color  # Top
        background[hold_y+box_size-border_thickness:hold_y+box_size, hold_x:hold_x+box_size] = border_color  # Bottom
        
        # Draw vertical lines
        background[hold_y:hold_y+box_size, hold_x:hold_x+border_thickness] = border_color  # Left
        background[hold_y:hold_y+box_size, hold_x+box_size-border_thickness:hold_x+box_size] = border_color  # Right
        
        return background

    def _create_grid_template(self):
        """Create grid lines template including edges"""
        # Calculate exact dimensions needed for the board region
        board_height = BOARD_HEIGHT * BLOCK_SIZE + 2  # +2 for borders
        board_width = BOARD_WIDTH * BLOCK_SIZE + 2   # +2 for borders
        
        grid = torch.zeros((board_height, board_width, 3),
                          dtype=torch.uint8, 
                          device=self.device)
        
        # Add grid lines (dark gray)
        grid[::BLOCK_SIZE, :] = 40
        grid[:, ::BLOCK_SIZE] = 40
        
        # Add consistent border lines
        border_intensity = 60
        grid[0:2, :] = border_intensity  # Top border
        grid[-2:, :] = border_intensity  # Bottom border
        grid[:, 0:2] = border_intensity  # Left border
        grid[:, -2:] = border_intensity  # Right border
        
        return grid

    def _create_piece_templates(self):
        """Pre-compute piece templates for each color"""
        templates = {}
        for piece_type, piece_id in PIECE_IDS.items():
            # Create block template with main color
            block = torch.zeros((BLOCK_SIZE, BLOCK_SIZE, 3), 
                              dtype=torch.uint8,
                              device=self.device)
            color = COLORS[piece_id]
            # Convert RGB color to BGR
            color = color[::-1]
            
            # Fill outer region with main color
            block[:, :] = torch.tensor(color, 
                                     dtype=torch.uint8,
                                     device=self.device)
            
            # Fill inner region with secondary color
            sec_color = SECONDARY_COLORS[piece_id]
            sec_color = sec_color[::-1]
            inner_margin = 2
            block[inner_margin:-inner_margin, 
                 inner_margin:-inner_margin] = torch.tensor(sec_color,
                                                          dtype=torch.uint8,
                                                          device=self.device)
            
            templates[piece_id] = block
            
        return templates

    def render_held_piece(self, held_piece, output):
        """Render held piece in the left panel with proper centering"""
        if held_piece:
            piece_id = PIECE_IDS[held_piece['type']]
            shape = held_piece['shape'][0]  # Use first rotation
            
            # Calculate box dimensions
            box_size = 4 * BLOCK_SIZE
            hold_x = self.side_panel_left + 50  # Moved further left
            hold_y = self.board_y + 70
            
            # Calculate piece dimensions
            piece_height = len(shape)
            piece_width = len(shape[0])
            
            # Get piece specific offsets
            offset = self.piece_offsets[held_piece['type']]
            
            # Calculate centering offsets (convert to integers)
            preview_block_size = int(BLOCK_SIZE * 0.8)
            center_x = int(hold_x + (box_size - piece_width * preview_block_size) / 2)
            center_y = int(hold_y + (box_size - piece_height * preview_block_size) / 2)
            
            # Apply piece-specific adjustments (convert to integers)
            center_x += int(offset['x'] * preview_block_size)
            center_y += int(offset['y'] * preview_block_size)
            
            # Render piece with calculated position
            self._render_piece_preview(output, shape, piece_id, center_x, center_y)

    def render_next_pieces(self, next_pieces, output):
        """Render next pieces preview in the right panel"""
        for i, piece in enumerate(next_pieces[:5]):  # Show up to 5 next pieces
            piece_id = PIECE_IDS[piece['type']]
            shape = piece['shape'][0]  # Use first rotation
            
            # Calculate position for each next piece
            next_x = self.side_panel_right + 10
            next_y = self.board_y + 60 + i * 60
            
            self._render_piece_preview(output, shape, piece_id, next_x, next_y)

    def _render_piece_preview(self, output, shape, piece_id, x, y):
        """Optimized piece preview rendering using tensor operations"""
        template = self.piece_templates[piece_id]
        preview_block_size = int(BLOCK_SIZE * 0.8)
        
        # Convert shape to tensor if not already
        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(shape, device=self.device)
            
        # Get piece positions
        positions = torch.nonzero(shape)
        if len(positions) > 0:
            rows, cols = positions[:, 0], positions[:, 1]
            
            # Calculate preview positions
            px = x + cols * preview_block_size
            py = y + rows * preview_block_size
            
            # Create resized template once
            preview_template = F.interpolate(
                template.permute(2,0,1).unsqueeze(0).float(),
                size=(preview_block_size, preview_block_size),
                mode='nearest'
            ).squeeze(0).permute(1,2,0).to(torch.uint8)
            
            # Draw preview blocks
            for px_i, py_i in zip(px, py):
                x1, y1 = px_i.item(), py_i.item()
                if (y1 < output.shape[0] - preview_block_size and 
                    x1 < output.shape[1] - preview_block_size):
                    output[
                        y1:y1+preview_block_size,
                        x1:x1+preview_block_size
                    ] = preview_template

    def render_board(self, board_state, current_piece, position, held_piece=None, next_pieces=None):
        """Render the full game state to a tensor"""
        # Reset output tensor
        self.output_tensor.zero_()
        
        # Copy static background with grid
        self.output_tensor.copy_(self.static_background)
        
        # Render board state using tensor operations
        with torch.amp.autocast(device_type='cuda'):
            # Get non-zero positions in board
            positions = torch.nonzero(board_state)
            if len(positions) > 0:
                rows, cols = positions[:, 0], positions[:, 1]
                piece_ids = board_state[rows, cols]
                
                # Calculate pixel coordinates
                px = self.board_x + cols * BLOCK_SIZE
                py = self.board_y + rows * BLOCK_SIZE
                
                # Draw blocks efficiently using tensor operations
                for idx, (piece_id, x, y) in enumerate(zip(piece_ids, px, py)):
                    template = self.piece_templates[piece_id.item()]
                    self.output_tensor[
                        y:y+BLOCK_SIZE,
                        x:x+BLOCK_SIZE
                    ] = template
        
            # Render current piece
            if current_piece:
                shape = torch.tensor(
                    current_piece['shape'][position[2]],
                    device=self.device
                )
                piece_id = PIECE_IDS[current_piece['type']]
                template = self.piece_templates[piece_id]
                
                # Get piece positions
                piece_pos = torch.nonzero(shape)
                if len(piece_pos) > 0:
                    # Fix: Correct tuple unpacking
                    rows, cols = piece_pos[:, 0], piece_pos[:, 1]
                    
                    # Calculate board positions
                    board_x = position[1] + cols
                    board_y = position[0] + rows
                    
                    # Filter valid positions
                    valid_mask = board_y >= 0
                    board_x = board_x[valid_mask]
                    board_y = board_y[valid_mask]
                    
                    # Calculate pixel coordinates
                    px = self.board_x + board_x * BLOCK_SIZE
                    py = self.board_y + board_y * BLOCK_SIZE
                    
                    # Draw piece blocks
                    for x, y in zip(px, py):
                        self.output_tensor[
                            y:y+BLOCK_SIZE,
                            x:x+BLOCK_SIZE
                        ] = template
        
        # Render UI elements using existing methods
        self.render_held_piece(held_piece, self.output_tensor)
        self.render_next_pieces(next_pieces, self.output_tensor)
        
        return self.output_tensor

    def _draw_block(self, target, x, y, template):
        """Draw a block template at the given board coordinates"""
        start_x = self.board_x + x * BLOCK_SIZE
        start_y = self.board_y + y * BLOCK_SIZE
        end_x = start_x + BLOCK_SIZE
        end_y = start_y + BLOCK_SIZE
        
        target[start_y:end_y, start_x:end_x] = template

# Modify TetrisEnv class to use TorchRenderer
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

        # Replace pygame screen with torch renderer
        self.renderer = TorchRenderer(SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE)
        self.render_delay = 100

        # Set up streams for async pruning
        self.cuda_streams = [torch.cuda.Stream() for _ in range(4)]

        self.rotation_results = [
            {
                'positions': torch.zeros((BOARD_WIDTH * 2, 2), dtype=torch.int32, device='cuda'),
                'scores': torch.zeros(BOARD_WIDTH * 2, dtype=torch.float32, device='cuda'),
                'valid_mask': torch.zeros(BOARD_WIDTH * 2, dtype=torch.bool, device='cuda')
            }
            for _ in range(4)
        ]

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
            if (time_delta > self.fall_speed * 1000):
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

    def get_state(self) -> torch.Tensor:
        """Get game state as visual representation"""
        # Render using torch renderer with all game elements
        board_tensor = self.renderer.render_board(
            self.game_state.board,
            self.game_state.current_piece,
            [self.game_state.position[0], 
             self.game_state.position[1],
             self.game_state.rotation_index],
            held_piece=self.game_state.hold_piece,
            next_pieces=self.game_state.next_pieces
        )
        
        if self.display_manager:
            self.display_manager.update(board_tensor)
            
        # Convert to grayscale using tensor operations
        return torch.mean(board_tensor.float(), dim=2).to(torch.uint8)

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

    def simulate_drop(self, board, position, piece_shape):
        """Simulate dropping a piece and return resulting board state"""
        # Create a copy of the board for simulation
        sim_board = board.clone()
        
        # Drop the piece until it hits something
        while position[0] < BOARD_HEIGHT:
            if not self.game_state.valid_position(piece=piece_shape, adj_y=1):
                break
            position[0] += 1
            
        # Place the piece on the simulated board
        for y, row in enumerate(piece_shape):
            for x, cell in enumerate(row):
                if cell:
                    new_y = position[0] + y
                    new_x = position[1] + x
                    if 0 <= new_y < BOARD_HEIGHT and 0 <= new_x < BOARD_WIDTH:
                        sim_board[new_y][new_x] = 1
                        
        return sim_board

    def evaluate_position(self, board):
        """Evaluate a board position using heuristics"""
        heights = calculate_heights(board, BOARD_HEIGHT)
        holes = count_holes(board, heights, BOARD_HEIGHT, BOARD_WIDTH)
        bumpiness = torch.sum(torch.abs(heights[:-1] - heights[1:]))
        
        # Check for line clears
        lines = 0
        for row in range(BOARD_HEIGHT):
            if torch.all(board[row] != 0):
                lines += 1
                
        # Weight the features
        height_weight = -0.51
        hole_weight = -0.36
        bumpiness_weight = -0.18
        line_weight = 0.76
        
        return (torch.max(heights).item() * height_weight + 
                holes * hole_weight + 
                bumpiness.item() * bumpiness_weight + 
                lines * line_weight)

    def get_best_action(self) -> int:
        """Get best action using cached target position"""
        # Convert piece shapes to tensor first
        piece_shapes = torch.tensor(self.game_state.current_piece['shape'], 
                                dtype=torch.int8, 
                                device='cuda')
        
        action, _, has_target, target_x, target_rot = find_best_move(
            self.game_state.board,
            piece_shapes,
            self.game_state.position,
            self.game_state.rotation_index,
            BOARD_WIDTH, BOARD_HEIGHT,
            self.game_state.has_target,
            self.game_state.target_x,
            self.game_state.target_rot,
            self.cuda_streams,
            self.rotation_results
        )
        
        # Update cached target
        self.game_state.has_target = has_target
        self.game_state.target_x = target_x
        self.game_state.target_rot = target_rot
        
        return action

    def find_best_landing_position(self, piece_shape: torch.Tensor, curr_position: List[int], 
                                 board: torch.Tensor) -> Tuple[torch.Tensor, int, int, float]:
        """Optimized best position finding using batch operations"""
        # Pre-allocate tensors for batch operations
        max_positions = (BOARD_WIDTH + 4) * 4  # All x positions * rotations
        test_boards = torch.zeros((max_positions, BOARD_HEIGHT, BOARD_WIDTH),
                                dtype=torch.int8, device='cuda')
        test_positions = torch.zeros((max_positions, 2), dtype=torch.int32, device='cuda')
        test_pieces = torch.zeros((max_positions, 4, 4), dtype=torch.int8, device='cuda')
        
        # Generate all possible positions and rotations
        valid_count = 0
        for rotation in range(4):
            rotated_shape = torch.tensor(piece_shape[rotation], device='cuda')
            
            for x in range(-2, BOARD_WIDTH + 2):
                # Quick check if position could be valid
                if (x + rotated_shape.size(1) <= 0 or 
                    x >= BOARD_WIDTH):
                    continue
                
                test_position = [0, x]
                if validate_move(board, rotated_shape, test_position):
                    test_positions[valid_count] = torch.tensor(test_position)
                    test_pieces[valid_count, :rotated_shape.size(0), 
                              :rotated_shape.size(1)] = rotated_shape
                    valid_count += 1
        
        if valid_count == 0:
            return None, curr_position[1], 0, float('-inf')
            
        # Simulate drops in batch
        valid_boards, final_y = simulate_drops_batch(
            board,
            test_positions[:valid_count],
            test_pieces[:valid_count]
        )
        
        # Calculate heights for all positions
        heights = calculate_heights_batch(valid_boards)
        
        # Evaluate all positions in parallel
        scores = evaluate_positions_batch(valid_boards, heights)
        
        # Find best position
        best_idx = torch.argmax(scores)
        best_score = scores[best_idx]
        best_position = test_positions[best_idx]
        best_board = valid_boards[best_idx]
        
        return (best_board, int(best_position[1]), 
                self._get_rotation_from_piece(test_pieces[best_idx]),
                float(best_score))
    
    def _get_rotation_from_piece(self, piece: torch.Tensor) -> int:
        """Determine rotation index from piece shape by comparing padded versions"""
        # Get piece dimensions
        max_size = 4  # Maximum piece size is 4x4
        padded_piece = torch.zeros((max_size, max_size), dtype=torch.int8, device='cuda')
        # Copy the actual piece into the padded tensor
        padded_piece[:piece.size(0), :piece.size(1)] = piece
        
        # Compare with each possible rotation
        for i, shape in enumerate(self.game_state.current_piece['shape']):
            # Create padded version of the shape
            padded_shape = torch.zeros((max_size, max_size), dtype=torch.int8, device='cuda')
            shape_tensor = torch.tensor(shape, device='cuda')
            padded_shape[:shape_tensor.size(0), :shape_tensor.size(1)] = shape_tensor
            
            if torch.all(padded_piece == padded_shape):
                return i
        return 0

    def evaluate_position_detailed(self, board: torch.Tensor) -> float:
        """Use batch evaluation for single position"""
        return float(evaluate_positions_batch(
            board.unsqueeze(0),
            calculate_heights(board, BOARD_HEIGHT).unsqueeze(0)
        )[0].item())

    def find_best_landing_position(self, piece_shape: torch.Tensor, curr_position: List[int],
                                 board: torch.Tensor) -> Tuple[torch.Tensor, int, int, float]:
        """Optimized position finding using CUDA streams and batching"""
        # Create CUDA streams for parallel processing
        streams = [torch.cuda.Stream() for _ in range(4)]
        
        # Pre-allocate tensors for all possible positions
        max_positions = (BOARD_WIDTH + 4) * 4
        all_positions = torch.zeros((max_positions, 2), dtype=torch.int32, device='cuda')
        all_pieces = torch.zeros((max_positions, 4, 4), dtype=torch.int8, device='cuda')
        all_rotations = torch.zeros(max_positions, dtype=torch.int32, device='cuda')
        
        # Generate position candidates more efficiently
        valid_count = 0
        x_range = torch.arange(-2, BOARD_WIDTH + 2, device='cuda')
        
        # Quick pruning - only consider positions that could be valid
        board_profile = torch.any(board != 0, dim=0)  # Get column occupancy
        valid_cols = torch.where(~board_profile)[0]  # Find empty columns
        if len(valid_cols) > 0:
            min_x = max(-2, valid_cols[0].item() - 2)
            max_x = min(BOARD_WIDTH + 2, valid_cols[-1].item() + 3)
            x_range = x_range[min_x+2:max_x+2]
        
        futures = []
        for rotation, stream in zip(range(4), streams):
            with torch.cuda.stream(stream):
                rotated_shape = torch.tensor(piece_shape[rotation], device='cuda')
                
                # Generate all x positions for this rotation
                positions = torch.stack([
                    torch.zeros_like(x_range),
                    x_range
                ], dim=1)
                
                # Batch validate all positions
                valid_mask = batch_validate_moves(
                    board,
                    rotated_shape.unsqueeze(0).expand(len(x_range), -1, -1),
                    positions
                )
                
                # Record valid positions
                valid_idx = torch.where(valid_mask)[0]
                if len(valid_idx) > 0:
                    count = len(valid_idx)
                    slice_end = valid_count + count
                    all_positions[valid_count:slice_end] = positions[valid_idx]
                    all_pieces[valid_count:slice_end, :rotated_shape.size(0), 
                              :rotated_shape.size(1)] = rotated_shape
                    all_rotations[valid_count:slice_end] = rotation
                    valid_count += count
                    
                futures.append(stream.record_event())
        
        # Synchronize streams
        for future in futures:
            future.wait()
            
        if valid_count == 0:
            return None, curr_position[1], 0, float('-inf')
            
        # Process valid positions in batches
        batch_size = 128  # Adjust based on GPU memory
        best_score = float('-inf')
        best_board = None
        best_x = curr_position[1]
        best_rotation = 0
        
        for i in range(0, valid_count, batch_size):
            end_idx = min(i + batch_size, valid_count)
            
            # Simulate drops for batch
            batch_boards, batch_y = simulate_drops_batch(
                board,
                all_positions[i:end_idx],
                all_pieces[i:end_idx]
            )
            
            # Calculate heights and evaluate positions
            heights = calculate_heights_batch(batch_boards)
            scores = evaluate_positions_batch(batch_boards, heights)
            
            # Update best position if better found
            max_score, max_idx = torch.max(scores, dim=0)
            if max_score > best_score:
                best_score = max_score
                best_board = batch_boards[max_idx]
                best_x = int(all_positions[i + max_idx, 1].item())
                best_rotation = int(all_rotations[i + max_idx].item())
        
        return best_board, best_x, best_rotation, float(best_score)

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
def calculate_heights_batch(boards: torch.Tensor) -> torch.Tensor:
    """Calculate heights for batch of boards
    
    Args:
        boards: Batch of boards [B, H, W]
        
    Returns:
        Heights for each column [B, W]
    """
    batch_size = boards.size(0)
    heights = torch.zeros((batch_size, boards.size(2)), 
                         dtype=torch.int32, device='cuda')
    
    # Use our calc heights on each board
    for b in range(batch_size):
        heights[b] = calculate_heights(boards[b], boards.size(1))
    
    return heights

@torch.jit.script
def calculate_reward(board, score: int, lines: int, weights, BOARD_HEIGHT: int, BOARD_WIDTH: int, actions_taken: int) -> float:
    """Calculate reward using GPU operations"""
    heights = calculate_heights(board, BOARD_HEIGHT)
    holes = torch.tensor(float(count_holes(board, heights, BOARD_HEIGHT, BOARD_WIDTH)), device='cuda')
    bumpiness = torch.tensor(float(torch.sum(torch.abs(heights[:-1] - heights[1:]).float()).item()), device='cuda')
    fill_reward = torch.tensor(float(calc_fill_reward(board)), device='cuda')
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

@torch.jit.script
def simulate_drops_batch(board: torch.Tensor, positions: torch.Tensor, 
                        pieces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Simulate dropping multiple pieces in parallel"""
    batch_size = positions.size(0)
    # Create new boards with proper copying
    boards = board.clone().expand(batch_size, -1, -1)  # Changed from expand+clone to clone+expand
    curr_positions = positions.clone()
    
    # Drop pieces until they can't move down
    while True:
        next_y = curr_positions[:, 0] + 1
        valid = torch.ones(batch_size, dtype=torch.bool, device='cuda')
        
        for b in range(batch_size):
            if next_y[b] >= board.size(0):
                valid[b] = False
                continue
                
            piece = pieces[b]
            pos = curr_positions[b]
            
            if not validate_move(board, piece, [int(pos[0]), int(pos[1])], 0, 1):
                valid[b] = False
        
        if not torch.any(valid):
            break
            
        curr_positions[valid, 0] = next_y[valid]
    
    # Place pieces on boards
    for b in range(batch_size):
        piece = pieces[b]
        pos = curr_positions[b]
        y, x = int(pos[0]), int(pos[1])
                
        piece_mask = piece != 0
        if torch.any(piece_mask):  # Only process if piece has content
            board_slice = boards[b, y:y+piece.size(0), x:x+piece.size(1)]
            if board_slice.size() == piece.size():
                board_slice[piece_mask] = piece[piece_mask]
    
    
    return boards, curr_positions[:, 0]

@torch.jit.script
def prune_positions(board: torch.Tensor, piece_shape: torch.Tensor, 
                    x_positions: torch.Tensor, y_position: int) -> torch.Tensor:
    """More strictly prune invalid positions"""
    piece_width = piece_shape.size(1)
    piece_height = piece_shape.size(0)
    board_width = board.size(1)
    
    # Stricter bounds check - ensure full piece is within board
    valid = (x_positions >= 0) & (x_positions <= board_width - piece_width)
    
    if not torch.any(valid):
        return torch.zeros_like(x_positions, dtype=torch.bool, device='cuda')
        
    # Check each candidate position
    valid_mask = torch.zeros_like(x_positions, dtype=torch.bool, device='cuda')
    
    for i, x in enumerate(x_positions):
        if not valid[i]:
            continue
            
        # Get board region where piece would be
        x_int = int(x.item())
        board_region = board[y_position:y_position+piece_height, 
                           x_int:x_int+piece_width]
                           
        # Only valid if no collision
        if board_region.size() == piece_shape.size():
            collision = torch.any((board_region != 0) & (piece_shape != 0))
            valid_mask[i] = not collision
            
    return valid_mask

def find_best_move(board: torch.Tensor, piece_shapes: torch.Tensor,
                          position: List[int], rotation_index: int,
                          BOARD_WIDTH: int, BOARD_HEIGHT: int,
                          has_target: bool, target_x: Optional[int],
                          target_rot: Optional[int],
                          streams: List[torch.cuda.Stream],
                          rotation_results: List[Dict[str, torch.Tensor]]) -> Tuple[int, float, bool, Optional[int], Optional[int]]:
    """Parallelized version of find_best_move using CUDA streams"""
    curr_x = position[1]
    curr_y = position[0] 
    curr_rot = rotation_index

    # Handle existing target
    if has_target and target_x is not None and target_rot is not None:
        if curr_rot != target_rot:
            rot_diff = (target_rot - curr_rot) % 4
            return (1 if rot_diff <= 2 else 2, 0.0, True, target_x, target_rot)
        elif curr_x < target_x:
            return (5, 0.0, True, target_x, target_rot)
        elif curr_x > target_x:
            return (4, 0.0, True, target_x, target_rot)
        return (0, 0.0, False, None, None)

    # Emergency drop check
    if curr_y < 4 and torch.any(board[0:8].sum(dim=1) > BOARD_WIDTH-2):
        return (0, 0.0, False, None, None)

    # Launch parallel evaluations for each rotation
    for rot in range(4):
        with torch.cuda.stream(streams[rot]):
            rot_shape = piece_shapes[rot]
            results = rotation_results[rot]
            
            # Generate and prune positions 
            x_positions = torch.arange(-2, BOARD_WIDTH + 2, device='cuda')
            valid_x = prune_positions(board, rot_shape, x_positions, curr_y)
            results['valid_mask'][:len(valid_x)] = valid_x
            
            if torch.any(valid_x):
                test_x = x_positions[valid_x]
                batch_size = test_x.size(0)
                
                # Setup positions
                positions = results['positions'][:batch_size]
                positions[:, 0] = curr_y
                positions[:, 1] = test_x
                
                # Create pieces tensor
                pieces = rot_shape.unsqueeze(0).expand(batch_size, -1, -1)
                
                # Simulate drops
                drop_boards, drop_y = simulate_drops_batch(board, positions, pieces)
                
                # Score positions
                heights = calculate_heights_batch(drop_boards)
                scores = evaluate_positions_batch(drop_boards, heights)
                
                # Apply penalties
                landing_heights = BOARD_HEIGHT - drop_y
                height_penalties = torch.clamp(landing_heights - torch.max(heights), min=0) * 0.5
                distance_penalties = torch.abs(positions[:, 1] - curr_x).float() * 0.1
                results['scores'][:batch_size] = scores - height_penalties - distance_penalties
            
    # Synchronize all streams
    for stream in streams:
        stream.synchronize()

    # Find best result across all rotations
    best_score = -float('inf')
    best_x = curr_x
    best_rot = curr_rot
    
    for rot in range(4):
        results = rotation_results[rot]
        valid_mask = results['valid_mask']
        if torch.any(valid_mask):
            scores = results['scores'][valid_mask]
            positions = results['positions'][valid_mask]
            
            max_score, max_idx = torch.max(scores, dim=0)
            if max_score > best_score:
                best_score = float(max_score)
                best_x = int(positions[max_idx, 1].item())
                best_rot = rot

    # Return movement command and cache target
    if best_score > -float('inf'):
        if curr_rot != best_rot:
            rot_diff = (best_rot - curr_rot) % 4
            return (1 if rot_diff <= 2 else 2, best_score, True, best_x, best_rot)
        elif curr_x < best_x:
            return (5, best_score, True, best_x, best_rot)
        elif curr_x > best_x:
            return (4, best_score, True, best_x, best_rot)
        return (0, best_score, False, None, None)
            
    return (6, 0.0, False, None, None)  # Hold piece if no good positions