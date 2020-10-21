from keras.datasets import mnist
import imageio


if __name__ == '__main__':
    (X_train, _), (_, _) = mnist.load_data()
    imageio.imsave('./result/mnist_image1.jpg', X_train[0])
