import pygame
import numpy as np


BLACK = ( 0, 0, 0)
WHITE = ( 255, 255, 255)
GREEN = ( 65, 74, 0)
YELLOW = (50, 50, 50)
RED = ( 255, 0, 0)
BLUE = ( 0, 0, 255 )

WIDTH = 500
HEIGHT = 500
WINDOWS_SIZE = (WIDTH, HEIGHT)

def gui_position(positions, width, height, size):

    return [positions[0] * width / size, positions[1] * height / size]

class Renderer:

    def __init__(self, size, town_map):


        self.size_ = size
        self.WIDTH_ = WIDTH
        self.HEIGHT_ = HEIGHT
        self.map_ = town_map
        self.car_position_ = [0, 0]
        self.destination_position_ = [0, 0]
        self.running_ = False

        self.update_map(town_map)

        # The clock will be used to control how fast the screen updates
        self.TICK_RATE_ = 60
        self.clock_ = pygame.time.Clock()

        pygame.init()



    def render(self):

        self.screen = pygame.display.set_mode(WINDOWS_SIZE)
        pygame.display.set_caption("My First Game")
        self.screen.fill(GREEN)

        for x in range(self.size_):
            pygame.draw.line(self.screen, RED, [x * self.WIDTH_ / self.size_, 0], [x * self.WIDTH_ / self.size_, self.HEIGHT_], 1)
            pygame.draw.line(self.screen, RED, [0, x * self.WIDTH_ / self.size_], [self.WIDTH_, x * self.WIDTH_ / self.size_], 1)

            if (self.car_position_ == self.destination_position_):
                pygame.draw.rect(self.screen, (230, 230, 0), gui_position(self.destination_position_, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)
            else:
                pygame.draw.rect(self.screen, BLUE, gui_position(self.car_position_, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)
                pygame.draw.rect(self.screen, YELLOW, gui_position(self.destination_position_, self.WIDTH_, self.HEIGHT_, self.size_) + [self.WIDTH_ / self.size_, self.WIDTH_ / self.size_], 0)

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
                if event.key == pygame.K_LEFT and self.car_position_[0] > 0:
                    self.car_position_[0] -= 1
                elif event.key == pygame.K_RIGHT and self.car_position_[0] < self.size_ - 1:
                    self.car_position_[0] += 1
                elif event.key == pygame.K_UP and self.car_position_[1] > 0:
                    self.car_position_[1] -= 1
                elif event.key == pygame.K_DOWN and self.car_position_[1] < self.size_ - 1:
                    self.car_position_[1] += 1
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
         

        self.car_position_ = [0, 0]
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


    def update_map(self, town_map):
        for y in range(self.size_):
            for x in range(self.size_):
                if town_map[y, x] == 1:
                    self.car_position_ = [x, y]
                elif town_map[y, x] == 2:
                    self.destination_position_ = [x, y]

    def policy(self):
        pass

