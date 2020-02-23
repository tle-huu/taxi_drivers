import pygame
import numpy as np


BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 65, 74, 0)
YELLOW = (200, 230, 50)
RED = ( 255, 0, 0)
BLUE = ( 0, 0, 255 )

WIDTH = 1000
HEIGHT = 1000
WINDOWS_SIZE = (WIDTH, HEIGHT)

DIRECTION = {"RIGHT": 0, "DOWN": 1, "LEFT": 2, "UP": 3, "IDLE": 4}

def gui_position(positions, width, height, size):

    return [positions['x'] * width / size, positions['y'] * height / size]

class Renderer:

    def __init__(self, size, town_map, number_of_cars):


        self.size_ = size
        self.WIDTH_ = WIDTH
        self.HEIGHT_ = HEIGHT
        self.map_ = town_map
        self.number_of_cars = number_of_cars
        self.cars_positions = None

        self.destination_position_ = {'x': -10, 'y': -10}
        self.car_position_ = {'x': -1, 'y': -1}
        self.running_ = False

        # The clock will be used to control how fast the screen updates
        self.TICK_RATE_ = 60
        self.clock_ = pygame.time.Clock()

        pygame.init()

    def draw_arrow(self, x, y, direction):

        SQUARE_SIZE = self.WIDTH_ / self.size_

        ARROW_LENGTH = int(SQUARE_SIZE / 8)
        ARROW_DEMI_LENGTH = int(ARROW_LENGTH / 2)

        def rotate(x, y):
            return y, -x

        x = x + int(SQUARE_SIZE / 2)
        y = y + int(SQUARE_SIZE / 2)

        force = int(SQUARE_SIZE / 4 )

        droite = [force, 0]
        haut = [0, -ARROW_DEMI_LENGTH]
        diag_bas = [ARROW_DEMI_LENGTH, ARROW_DEMI_LENGTH]
        diag_haut = [-ARROW_DEMI_LENGTH, ARROW_DEMI_LENGTH]
        bas = [0, -ARROW_DEMI_LENGTH]

        vecs = [droite, haut, diag_bas, diag_haut, bas]

        for i in range(len(vecs)):
            a, b = vecs[i]
            for _ in range(direction):
                a, b = rotate(a, b)
            vecs[i] = [a, b]


        points = [[x, y]]
        current = points[-1]
        for v in vecs:
            points.append([current[0] + v[0], current[1] + v[1]])
            current = points[-1]


        pygame.draw.polygon(self.screen, BLACK, points)
    # pygame.draw.polygon(window, (0, 0, 0), ((0, 100), (0, 200), (200, 200), (200, 300), (300, 150), (200, 0), (200, 100)))


    def render(self):

        self.screen = pygame.display.set_mode(WINDOWS_SIZE)
        pygame.display.set_caption("My First Game")
        self.screen.fill(GREEN)


        for x in range(self.size_):
            for y in range(self.size_):

                current = {'x': y, 'y': x}

                if self.map_[x, y] == -1:
                    pygame.draw.rect(self.screen, BLACK, gui_position(current, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)
                if self.map_[x, y] == 50000:
                    pygame.draw.rect(self.screen, RED, gui_position(current, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)

                self.draw_arrow(*gui_position(current, self.WIDTH_, self.HEIGHT_, self.size_), DIRECTION['LEFT'])
                self.draw_arrow(*gui_position(current, self.WIDTH_, self.HEIGHT_, self.size_), DIRECTION['UP'])
                self.draw_arrow(*gui_position(current, self.WIDTH_, self.HEIGHT_, self.size_), DIRECTION['RIGHT'])
                self.draw_arrow(*gui_position(current, self.WIDTH_, self.HEIGHT_, self.size_), DIRECTION['DOWN'])

        for x in range(self.size_):
            pygame.draw.line(self.screen, RED, [x * self.WIDTH_ / self.size_, 0], [x * self.WIDTH_ / self.size_, self.HEIGHT_], 1)
            pygame.draw.line(self.screen, RED, [0, x * self.WIDTH_ / self.size_], [self.WIDTH_, x * self.WIDTH_ / self.size_], 1)

        # Draw destination
        pygame.draw.rect(self.screen, YELLOW, gui_position(self.destination_position_, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)

        for car_position in self.cars_positions:
            if (car_position == self.destination_position_):
                pygame.draw.rect(self.screen, (230, 230, 0), gui_position(self.destination_position_, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)
            else:
                pygame.draw.rect(self.screen, BLUE, gui_position(car_position, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)
        

        # Rerender
        pygame.display.flip()

    def reset(self):
        pass

    def close(self):
        pygame.quit()


    def events_loop(self):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.car_position_['x'] > 0:
                    self.car_position_['x'] -= 1
                elif event.key == pygame.K_RIGHT and self.car_position_['x'] < self.size_ - 1:
                    self.car_position_['x'] += 1
                elif event.key == pygame.K_UP and self.car_position_['y'] > 0:
                    self.car_position_['y'] -= 1
                elif event.key == pygame.K_DOWN and self.car_position_['y'] < self.size_ - 1:
                    self.car_position_['y'] += 1
            if event.type == pygame.QUIT: # If user clicked close
                  self.running_ = False # Flag that we are done so we exit this loop

    def start(self):

        self.running_ = True

        # town, self.car_position_, self.destination_position_ = tools.parser("map.txt")

        # Open a new window
        screen = pygame.display.set_mode(WINDOWS_SIZE)
        pygame.display.set_caption("TaxiDriver")

        # The loop will carry on until the user exit the game (e.g. clicks the close button).
        self.running_ = True
         

        # self.car_position_ = [0, 0]
        # -------- Main Program Loop -----------
        while self.running_:
            # --- Main event loop
            self.events_loop()

            # Rendering
            self.render()

            # Actions
            self.policy()
             
            self.clock_.tick(self.TICK_RATE_)

        self.close()

    ## !! These are changing the referense !!
    def set_cars_position(self, cars_positions):
        self.cars_positions = cars_positions

    def set_destination_position(self, new_destination_position):
        self.destination_position_ = new_destination_position


    def policy(self):
        pass

