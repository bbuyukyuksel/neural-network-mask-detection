import utils
import Parameters
import model
import numpy as np
import matplotlib.pyplot as plt

_, parameters = Parameters.load_last()

classes = [
    "without_mask",
    "with_mask",
]

Test_Image = r'tdataset\prepared\dev\1-0185.png'
real_image = utils.prepare_image(Test_Image, image_size=64)

image = real_image.copy()
image_flatten = image.reshape(1, -1).T
image = image_flatten/255.
p, _ = model.predict(image, np.array([[0]]), parameters)

plt.imshow(real_image, interpolation='nearest')
plt.title(f'predict: {classes[int(np.squeeze(p))]}')
plt.axis('off')
plt.show()
