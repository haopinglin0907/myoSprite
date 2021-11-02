# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:12:28 2021

@author: haopi
"""

import arcade
import random
import numpy as np

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Myo_Calibration"


class MyGame(arcade.Window):
    """ Our custom Window Class"""

    def __init__(self):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        self.length = 0
        self.label = 'Rest'
        self.status = 'Collecting gesture data'
        self.trainingSide = 'False'
        self.color = arcade.color.WHITE
        self.total_time = 0.0
        self.output = "00:00:00"
        arcade.set_background_color(arcade.color.AMAZON)

    def setup(self):
        """ Set up the game and initialize the variables. """
        self.length = 0
        self.label = 'Rest'
        self.trainingSide = 'False'
        self.total_time = 0.0
        self.color = arcade.color.WHITE


    def on_draw(self):
        """ Draw everything """
        arcade.start_render()

        # Output the timer text.
        arcade.draw_text(self.output,
                         SCREEN_WIDTH - 55, SCREEN_HEIGHT - 35,
                         arcade.color.WHITE, 20,
                         anchor_x="center")

        # Status
        arcade.draw_text(self.status,
                         SCREEN_WIDTH / 2, SCREEN_HEIGHT* 0.15 + 10,
                         arcade.color.WHITE, 20,
                         anchor_x="center")
        # Label
        arcade.draw_lrtb_rectangle_filled(SCREEN_WIDTH / 2 - 200,
                                          SCREEN_WIDTH / 2 -200 + (self.length % 600) / 600 * 400,
                                          SCREEN_HEIGHT* 0.10 + 29,
                                          SCREEN_HEIGHT* 0.15 - 39,
                                          arcade.color.BRIGHT_GREEN)

        arcade.draw_text(self.label, SCREEN_WIDTH / 2, SCREEN_HEIGHT* 0.10,
                         arcade.color.WHITE, 20, anchor_x="center")

        arcade.draw_lrtb_rectangle_outline(SCREEN_WIDTH / 2 - 200, SCREEN_WIDTH / 2 + 200, SCREEN_HEIGHT* 0.10 + 30,
                                           SCREEN_HEIGHT* 0.15 - 40, arcade.color.WHITE, border_width=3)



        # draw channel bars
        for channel in range(0, 8):
            arcade.draw_lrtb_rectangle_outline(100 + 75*channel, 150 + 75*channel, 450,
                                               150, arcade.color.WHITE, border_width=2)

            arcade.draw_lrtb_rectangle_filled(102 + 75*channel, 149 + 75*channel, 150 + (self.magnitude[channel] / 128)*300, 150, arcade.color.BRIGHT_GREEN)


    def on_update(self, delta_time):
        """
        All the logic to move, and the game logic goes here.
        """
        self.total_time += delta_time

        # Calculate minutes
        minutes = int(self.total_time) // 60

        # Calculate seconds by using a modulus (remainder)
        seconds = int(self.total_time) % 60

        # Calculate 100s of a second
        seconds_100s = int((self.total_time - seconds) * 100)

        # Figure out our output
        self.output = f"{minutes:02d}:{seconds:02d}:{seconds_100s:02d}"


def main():
    """ Main method """
    window = MyGame()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()