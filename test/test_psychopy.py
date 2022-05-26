from psychopy import visual, core, event
import time

# create window
# later adjust size to tablet!!!
win0 = visual.Window([800, 600], screen = 0, monitor = 'testMonitor',
                     fullscr=False, color=[0, 0, 0], units='pix')

gratingStimulus = visual.GratingStim(win=win0, tex='sin', units='pix', pos=(
    0.0, 0.0), size=800, sf=0.01, ori=0.0, phase=(0.0, 0.0))

win0.getMovieFrame(buffer='back')
startTime = time.time()
runTime = 2 # run stimulus for 15 seconds
images = []
while(time.time() - startTime < runTime):
 #   win0.getMovieFrame(buffer='back') # get every upcoming frame?
    gratingStimulus.setPhase(0.02, '+')
    # 1st parameter is for speed of drifting
    # 2nd parameter is for direction of drifint ('+': left to right)
    gratingStimulus.draw()
    win0.flip()
    img = win0.getMovieFrame()
    images.append(img)

    if len(event.getKeys()) > 0:
        break
    event.clearEvents()
print("Play time: ", time.time()-startTime)    
win0.close()
win0.saveMovieFrames(fileName='./testMovie.mp4') # save frames as video file
