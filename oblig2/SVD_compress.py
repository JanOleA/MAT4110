from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def plot_compression_ratio():
    """ Calculates and plots the compression ratio for an m x n image as a
    function of k
    """
    for m in [512, 1024, 2048]:
        for n in [1024]:
            k_max = min(m, n)
            k = np.linspace(25, k_max, 1000)
            cr = m*n/(k*(1 + m + n))
            plt.plot(k, cr, label = "m = {:d}, n = {:d}".format(m, n))

    plt.plot([25,k_max], [1,1], "r--", alpha = 0.5, label = "1")
    plt.legend()
    plt.title("Compression ratios")
    plt.xlabel("k")
    plt.ylabel("Compression ratio")
    plt.savefig("images/CRs.pdf")
    plt.show()

def SVD_size(U, S, V):
    """ Returns the number of values required to store a singular value
    decomposition
    """
    return U.size + S.size + V.size


def rgb2gray(im):
    """ Takes a color Pillow image and returns a 2d grayscale Numpy matrix
    with values scaled between 0 and 1
    """
    return np.matrix(im.convert("L"))/255


def SVD_compress(im, k):
    max_k = np.min(im.shape)
    if k > max_k:
        print("Chosen k is larger than smallest dimension, reducing to k =",
               max_k)
        k = max_k

    U, S, V = np.linalg.svd(im)
    U = U[:,:k]
    S = S[:k]
    V = V[:k,:]

    return U, S, V, k


def SVD_decompress(U, S, V):
    US = U @ np.diag(S)
    im = US @ V
    return im


plot_compression_ratio()


im = Image.open("chessboard.png")
im = rgb2gray(im)

im2 = Image.open("jellyfish.jpg")
im2 = rgb2gray(im2)

im3 = Image.open("new-york.jpg")
im3 = rgb2gray(im3)


image_list = [im, im2, im3]
image_names = ["chessboard", "jellyfish", "new-york"]
k_list = [int(1280*2**(-i)) for i in range(11)]
print(k_list)

for image, image_name in zip(image_list, image_names):
    plt.figure()
    plt.imshow(image, cmap="gray")
    plt.title("Original")
    fn = "images/{:s}-original.pdf".format(image_name)
    plt.savefig(fn)

    for k in k_list:
        U, S, V, k = SVD_compress(image, k)
        CR = image.size/SVD_size(U, S, V)
        im_compressed = SVD_decompress(U, S, V)
        plt.figure()
        plt.imshow(im_compressed, cmap="gray")
        plt.title("Compressed, k = {:d}, CR = {:f}".format(k, CR))
        fn = "images/{:s}-{:04d}.pdf".format(image_name, k)
        plt.savefig(fn, dpi = 600)

    U, S, V, k = SVD_compress(image, np.min(image.shape))
    plt.figure()
    cross_1 = np.argmin(np.abs(S - 1))
    print("First 5 singular values:", S[:5])
    print("Singular values cross 1 at k={:d}".format(cross_1))
    print("Singular values at cross:", S[cross_1-3:cross_1+2])
    plt.semilogy(S)
    plt.title("Singular values")
    plt.xlabel("index")
    fn = "images/{:s}-svs.pdf".format(image_name, k)
    plt.savefig(fn)

    plt.show()
