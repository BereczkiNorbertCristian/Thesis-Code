

import skvideo.io
import skvideo.datasets
videodata = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
print(videodata.shape)

