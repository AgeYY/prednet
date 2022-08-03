import numpy as np

label_id = (0, 1)
label = [[0, 1], [2, 3]]
print( *label )

label = np.array( [[0, 1], [2, 3]] )
print( label[:, label_id] )

