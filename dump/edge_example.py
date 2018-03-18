
from skimage import feature, data, io, filters

image = data.coins()

sobel_edges=filters.sobel(image)
canny_edges_1=feature.canny(image,sigma=1)
canny_edges_3=feature.canny(image,sigma=3)

io.imshow_collection([image,sobel_edges,canny_edges_1,canny_edges_3])
io.show()

