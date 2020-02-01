import numpy as np

def parser(file):

    width = -1
    height = -1
    mapp = None
    y = -1
    car_position = [0,0]
    goal_position = [-1, -1]
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            y += 1
            if width < 0:
                line = line.split(" ")
                width = int(line[0])
                height = int(line[1])

                mapp = np.zeros([height, width])
                y = 0
            else:
                for x in range(width):
                    if line[x] == 'X':
                        mapp[y, x] = 1
                        car_position = [x, y]
                    elif line[x] == 'G':
                        mapp[y, x] = 2
                        goal_position = [x, y]
    return mapp, car_position, goal_position