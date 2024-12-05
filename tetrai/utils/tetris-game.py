import pygame
import sys
sys.path.append('..')
from tetrai import TetrisEnv
import numpy as np

# Action mapping
ACTION_MAP = {
    pygame.K_ESCAPE: 'quit',
    pygame.K_UP: 1,     # Rotate Right
    pygame.K_LEFT: 5,   # Move Left
    pygame.K_RIGHT: 6,  # Move Right
    pygame.K_DOWN: 4,   # Move Down
    pygame.K_SPACE: 7,  # Force Down
    pygame.K_z: 2,      # Rotate Left
    pygame.K_c: 8       # Hold
}

def main():
    env = TetrisEnv.TetrisEnv()
    env.render_mode = True
    state = env.reset()
    done = False

    pygame.init()
    clock = pygame.time.Clock()

    while not done:
        action = 0  # Default action: Nothing

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in ACTION_MAP:
                    if ACTION_MAP[event.key] == 'quit':
                        done = True
                        break
                    else:
                        action = ACTION_MAP[event.key]

        if done:
            break

        next_state, reward, done = env.step(action)
        state = next_state

        env.render()
        clock.tick(60)  # Limit to 60 FPS

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()