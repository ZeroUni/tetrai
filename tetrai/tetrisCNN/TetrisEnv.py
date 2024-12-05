import pygame
import random
import numpy as np
from pygame import Surface
from typing import Tuple

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
        self.board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=int)
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
        if piece is None:
            piece = self.current_piece['shape'][self.rotation_index]
        for y, row in enumerate(piece):
            for x, cell in enumerate(row):
                if cell:
                    new_x = self.position[1] + x + adj_x
                    new_y = self.position[0] + y + adj_y
                    if new_x < 0 or new_x >= BOARD_WIDTH or new_y >= BOARD_HEIGHT:
                        return False
                    if new_y >= 0 and self.board[new_y][new_x]:
                        return False
        return True

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
        self.position = [0, BOARD_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2]
        self.rotation_index = 0
        if not self.valid_position():
            self.game_over = True
        # Up the score a bit
        self.score += 2
        self.can_hold = True
        self.is_landing = False
        self.lock_timer = 0

    def clear_lines(self):
        lines_before = self.lines_cleared
        self.board = np.array([row for row in self.board if not all(row)])
        cleared = BOARD_HEIGHT - len(self.board)
        self.lines_cleared += cleared
        match cleared:
            case 1:
                self.score += 40
            case 2:
                self.score += 100
            case 3:
                self.score += 300
            case 4:
                self.score += 1200
        if self.lines_cleared >= 10 * (lines_before // 10 + 1):
            self.score += 1000
        # Add empty rows on top
        while len(self.board) < BOARD_HEIGHT:
            self.board = np.vstack([np.zeros(BOARD_WIDTH, dtype=int), self.board])

class TetrisEnv:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('TetrisEnv')
        self.clock = pygame.time.Clock()
        self.game_state = GameState()
        self.fall_time = 0
        self.fall_speed = 0.5  # Seconds per fall
        self.last_move_time = pygame.time.get_ticks()
        self.render_mode = False  # Set to False to run without rendering

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
        self.game_state = GameState()
        self.fall_time = 0
        self.last_move_time = pygame.time.get_ticks()
        return self.get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
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
            reward -= 100  # Penalty for dying

        return self.get_state(), reward, done

    def apply_action(self, action: int):
        moved = False
        if action == 0:  # Nothing
            return
        elif action == 1:  # Rotate Right
            self.game_state.rotate_piece(direction=1)
            moved = True
        elif action == 2:  # Rotate Left
            self.game_state.rotate_piece(direction=-1)
            moved = True
        elif action == 3:  # Move Up
            if self.game_state.valid_position(adj_y=-1):
                self.game_state.position[0] -= 1
                moved = True
        elif action == 4:  # Move Down
            if self.game_state.valid_position(adj_y=1):
                self.game_state.position[0] += 1
                moved = True
        elif action == 5:  # Move Left
            if self.game_state.valid_position(adj_x=-1):
                self.game_state.position[1] -= 1
                moved = True
        elif action == 6:  # Move Right
            if self.game_state.valid_position(adj_x=1):
                self.game_state.position[1] += 1
                moved = True
        elif action == 7:  # Hard Drop
            while self.game_state.valid_position(adj_y=1):
                self.game_state.position[0] += 1
            self.game_state.lock_piece()
            self.game_state.is_landing = False
            self.game_state.lock_timer = 0
        elif action == 8:  # Hold
            self.game_state.hold_current_piece()
            moved = True

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

        if not self.render_mode:
            return

        self.render()
        self.clock.tick(60)

    def calculate_reward(self) -> float:
        reward = 0
        # Good rewards
        reward += self.game_state.score * 0.1  # Scoring
        reward += self.game_state.lines_cleared * 10  # Line clears
        # Add more rewards for complex moves if implemented

        # Bad rewards
        height = self.get_board_height()
        reward -= height * 0.5  # Penalize high stacks
        holes = self.count_holes()
        reward -= holes * 1  # Penalize holes

        return reward

    def get_board_height(self) -> int:
        for y in range(BOARD_HEIGHT):
            if any(self.game_state.board[y]):
                return BOARD_HEIGHT - y
        return 0

    def count_holes(self) -> int:
        holes = 0
        for x in range(BOARD_WIDTH):
            block_found = False
            for y in range(BOARD_HEIGHT):
                if self.game_state.board[y][x]:
                    block_found = True
                elif block_found and not self.game_state.board[y][x]:
                    holes += 1
        return holes

    def get_state(self) -> np.ndarray:
        # Render the board to a surface
        board_surface = Surface(SCREEN_WIDTH, SCREEN_HEIGHT)
        board_surface.fill(BLACK)

        # Draw game board in center, slightly above the bottom, with grid pattern
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                rect = pygame.Rect(self.board_x + x * BLOCK_SIZE, self.board_y + y * BLOCK_SIZE , BLOCK_SIZE, BLOCK_SIZE)
                pygame.draw.rect(board_surface, (40, 40, 40), rect, 1)
                if self.game_state.board[y][x]:
                    pygame.draw.rect(board_surface, COLORS[self.game_state.board[y][x]], rect)
                    pygame.draw.rect(board_surface, SECONDARY_COLORS[self.game_state.board[y][x]], 
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
                        pygame.draw.rect(board_surface, border_color, border)

                        rect = pygame.Rect(px + 2, py + 2, BLOCK_SIZE - 4, BLOCK_SIZE - 4)
                        pygame.draw.rect(board_surface, color, rect)

        # Draw side panel background
        side_panel_rect = pygame.Rect(
            self.side_panel_x_left - self.side_panel_width_left,
            self.board_y + 35,
            self.side_panel_width_left,
            140
        )

        pygame.draw.rect(board_surface, (30, 30, 30), side_panel_rect)

        # Draw "Hold" label
        self.draw_text("Held:", (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 40), board_surface)

        # Draw held piece
        if self.game_state.hold_piece:
            self.draw_small_piece(self.game_state.hold_piece, (self.side_panel_x_left - self.side_panel_width_left + 50, self.board_y + 110), board_surface)

        # Draw a little box around the hold piece
        pygame.draw.rect(board_surface, WHITE, (self.side_panel_x_left - self.side_panel_width_left + 10, self.board_y + 80, 4 * BLOCK_SIZE, 4 * BLOCK_SIZE), 1)

        # Draw "Next" label
        self.draw_text("Next", (self.side_panel_x_right + 20, self.board_y + 5), board_surface)

        # Draw next five pieces
        for i, piece in enumerate(self.game_state.next_pieces[:5]):
            self.draw_small_piece(piece, (self.side_panel_x_right + 40, self.board_y + 65 + i * 60), board_surface)

        # Convert the surface to a numpy array
        raw_str = pygame.image.tostring(board_surface, 'RGB')
        image = np.frombuffer(raw_str, dtype=np.uint8)
        image = image.reshape((BOARD_HEIGHT * BLOCK_SIZE, BOARD_WIDTH * BLOCK_SIZE, 3))
        image = np.mean(image, axis=2)  # Convert to grayscale
        image = image.astype(np.uint8)

        return image

    def render(self):
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