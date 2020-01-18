import numpy as np

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
    pass
    




if __name__ == '__main__':
    main()

    print_grid(grid)
