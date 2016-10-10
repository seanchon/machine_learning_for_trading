"""Creating NumPy arrays."""
import numpy as np


def test_run():
    # Empty array - filled with random values from memory
    print(np.empty(5))
    print(np.empty((5, 4)))

    # Array of 1s
    print(np.ones((5, 4), dtype=np.int))

    # Array of 0s
    print(np.zeros((5, 4), dtype=np.int))

    # Array of random numbers, uniformly sampled from [0.0, 1.0)
    print(np.random.random((5, 4)))

    # Sample numbers from a Gaussian (normal) distribution
    print(np.random.normal(size=(2, 3)))  # "standard normal" (mean=0, s.d. = 1)
    print(np.random.normal(50, 10, size=(2, 3)))  # change mean to 50 and s.d. to 10

    # Random integers
    print(np.random.randint(10))  # a single integer in [0, 10)
    print(np.random.randint(0, 10))  # same as above while explicitly specifying [0, 10)
    print(np.random.randint(0, 10, size=5))  # 5 random integers as a 1D array
    print(np.random.randint(0, 10, size=(2, 3)))  # 2x3 array of random integers


def test_run2():
    a = np.random.random((5, 4))  # 5x4 array of random numbers
    print(a)
    print(a.shape)
    print(a.shape[0])  # number of rows
    print(a.shape[1])  # number of columns
    print(len(a.shape))  # dimension of array
    print(a.size)  # number of elements in array
    print(a.dtype)  # element type


def test_run3():
    np.random.seed(693)  # seed the random number generator
    a = np.random.randint(0, 10, size=(5, 4))  # 5x4 random integers in [0, 10)
    print("Array:\n{}".format(a))

    # Sum of all elements
    print("Sum of all elements: {}".format(a.sum()))

    # Iterate over rows to compute sum of each column
    print("Sum of each column:\n{}".format(a.sum(axis=0)))

    # Iterate over column to compute sum of each row
    print("Sum of each row:\n{}".format(a.sum(axis=1)))

    # Statistics: min, max, mean (across rows, cols, and overall)
    print("Minimum of each column:\n{}".format(a.min(axis=0)))
    print("Maximum of each row:\n{}".format(a.max(axis=1)))
    print("Mean of all elements:\n{}".format(a.mean()))  # leave out axis arg


if __name__ == "__main__":
    test_run3()
