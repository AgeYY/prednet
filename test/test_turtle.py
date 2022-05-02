### using pillow to create videos seems easier
import turtle
import os

init_x = -250
turtlesize = (10, 3, 0)
speed = 1
forward = 500
shape = 'square'

running = True
FRAMES_PER_SECOND = 10

def draw():
    bob = turtle.Turtle()
    bob.shape(shape)
    bob.penup() # we only care about the shape

    bob.hideturtle()
    bob.setx(init_x)
    bob.showturtle()

    bob.turtlesize(*turtlesize)

    bob.speed(speed)

    bob.forward(forward)

    turtle.ontimer(stop, 500)
    #turtle.done()

def stop():
    global running

    running = False

class Recorder():
    def __init__(self, frames_per_second=10, save_dir_head='./kitti_data/raw/', save_dir_label='output/'):
        '''
        delta t is defined as 1 between each pair of consecutive frames. In this case, the equvilent speed of object is propotional to 1 / frames_per_second. This allows us to change the speed continuously
        '''
        self.running = True
        self.frames_per_second = frames_per_second
        self.save_dir_head=save_dir_head
        self.save_dir_label=save_dir_label

    def start(self, counter=[1]):
        save_dir = self.save_dir_head + self.save_dir_label
        if not os.path.exists(save_dir): os.mkdir(save_dir) # if doesn't exist, create the dir

        turtle.getcanvas().postscript(file =  save_dir + 'moving_bar{0:03d}.eps'.format(counter[0]))
        counter[0] += 1
        if self.running:
            turtle.ontimer(self.start, int(1000 / self.frames_per_second))

    def stop(self):
        self.running = False

    def draw(self, draw_func):
        turtle.ontimer(draw, 500)  # start the program (1/2 second leader)
        #self.running = False


recorder = Recorder(save_dir_label='moving_bar/')

recorder.start()  # start the recording

recorder.draw(draw)
#turtle.ontimer(draw, 500)  # start the program (1/2 second leader)
#
turtle.done()

#recorder.stop()
