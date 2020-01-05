#This script was to test the plot display of picture
#because the 

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

img = mpimg.imread('./ahs.jpg')
imgplot = plt.imshow(img)
plt.show()