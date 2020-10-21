from keras.models import load_model
import numpy as np
import imageio


if __name__ == '__main__':
    model = load_model('GAN_model/GAN/generator_model_fixed1.h5')
    print(model.summary())
    results = model.predict(np.random.uniform(-1, 1, (20, 100)))
    for i in range(len(results)):
        imageio.imsave('./result/fixed/img {}.jpg'.format(i), ((results[i] + 1) * 127.5).astype(np.uint8))
