import numpy as np
import pygame

grid = np.zeros([4, 5])

grid[0][0] = 1
grid[-1][-1] = 2

PRINTING_DICT = {0: " ",  \
                 1: "C",  \
                 2: "D"   \
                 }

#  0 => roads
#  1 => starting position the car
#  2 => destination
def print_grid(grid):
    
    height, width = grid.shape

    print(grid)

    for i in range(width + 2):
        print("_", end='')
    print("")
    for i in range(height):
        print("|", end='')
        for j in range(width):
            print(PRINTING_DICT[grid[i][j]], end='')
        print("|", end='')
        print("")
    for i in range(width + 2):
        print("-", end='')
    print("")


def main():

    for y in range(height):
        for x in range(width):
            rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
            pygame.draw.rect(window, color, rect)
    # head
    x, y = snake[0]
    rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
    pygame.draw.rect(window, head_color, rect)

    # tail
    for x, y in snake[1:]:
        rect = pygame.Rect(x*block_size, y*block_size, block_size, block_size)
        pygame.draw.rect(window, tail_color, rect)    




if __name__ == '__main__':
    main()

    print_grid(grid)
