import numpy as np

def parser(file):

    width = -1
    height = -1
    mapp = None
    y = -1
    car_position = {'x': 0, 'y': 0}
    goal_position = {'x': 0, 'y': 0}
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            y += 1
            if width < 0:
                line = line.split(" ")
                width = int(line[0])
                height = int(line[1])

                mapp = np.zeros([height, width])
                y = -1
            else:
                for x in range(len(line)):
                    # if line[x] == 'X':
                    #     mapp[y, x] = 1
                    #     car_position = {'x': x, 'y': y}
                    if line[x] == 'G':
                        mapp[y, x] = 10000
                        goal_position = {'x': x, 'y': y}
                    if line[x] == 'S':
                        mapp[y, x] = 50000
                    elif line[x] == '#':
                        mapp[y, x] = -1

    return mapp, car_position, goal_position