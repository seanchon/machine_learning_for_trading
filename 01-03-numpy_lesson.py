"""Creating NumPy arrays."""
from lib.time_function import how_long
import numpy as np
import time


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


def test_run4():
    t1 = time.time()
    print("ML4T")
    t2 = time.time()
    print("The time taken by print() is {} seconds".format(t2 - t1))


def manual_mean(arr):
    """Compute mean (average) of all elements in a given 2D array."""
    sum = 0
    for i in xrange(0, arr.shape[0]):
        for j in xrange(0, arr.shape[1]):
            sum = sum + arr[i, j]

    return sum / arr.size


def numpy_mean(arr):
    """Compute mean (average) using NumPy."""
    return arr.mean()


def test_run5():
    nd1 = np.random.random((1000, 10000))  # use a sufficiently large array

    # Time the two functions, retrieving results and execution times
    res_manual, t_manual = how_long(manual_mean, nd1)
    res_numpy, t_numpy = how_long(numpy_mean, nd1)
    print("Manual: {:.6f} ({:.3f} secs.) vs. NumPy: {:.6f} ({:.3f} secs.)".format(res_manual, t_manual, res_numpy, t_numpy))

    # Make sure both give us the same answer (up to some precision)
    assert abs(res_manual - res_numpy) <= 10e-6, "Results are not equal!"

    # Compute speedup
    speedup = t_manual / t_numpy
    print("NumPy mean is {} times faster than manual for loops.".format(speedup))


def test_run6():
    a = np.random.rand(5, 4)
    print("Array:\n{}".format(a))

    # Accessing element at position (3, 2)
    element = a[3, 2]
    print(element)

    # Slicing
    # Elements in defined range
    print(a[0, 1:3])

    # Top left corner
    print(a[0:2, 0:2])

    # Note: Slice n:m:t specifies a range that starts at n, and stops before m, in steps of size t
    print(a[:, 0:3:2])


def test_run7():
    a = np.random.rand(5, 4)
    print("Array:\n{}".format(a))

    # Assigning a value to a particular location
    a[0, 0] = 1
    print("\nModified (replaced one element):\n{}".format(a))

    # Assigning a single value to an entire row
    a[2, :] = 1
    print("\nModified (replaced row with a single value):\n{}".format(a))

    # Assigning a list to a column in an array
    a[:, 3] = [1, 2, 3, 4, 5]
    print("\nModified (replaced column with a list):\n{}".format(a))


def test_run8():
    a = np.random.rand(5)
    print(a)

    # Accessing using a list of indices
    indices = np.array([1, 1, 2, 3])

    print(a[indices])


def test_run9():
    a = np.array([(20, 25, 10, 23, 26, 32, 10, 5, 0), (0, 2, 50, 20, 0, 1, 28, 5, 0)])
    print(a)

    # Calculating mean
    mean = a.mean()
    print(mean)

    # Masking (print values less than mean)
    print(a[a < mean])

    # Masking (replace values less than mean with mean value)
    a[a < mean] = mean
    print(a)


def test_run10():
    a = np.array([(1, 2, 3, 4, 5), (10, 20, 30, 40, 50)])
    print("Original array:\n{}".format(a))

    # Multiply by two
    print("\nMultiply by two:\n{}".format(a * 2))

    # Divide by two (using decimal to get float values instead of rounded-off ints)
    print("\nDivide by two:\n{}".format(a / 2.0))

    b = np.array([(100, 200, 300, 400, 500), (1, 2, 3, 4, 5)])
    print("\nOriginal array b:\n{}".format(b))

    # Add the two arrays
    print("\nAdd a + b:\n{}".format(a + b))

    # Multiply the two arrays
    print("\nMultiply a * b:\n{}".format(a * b))

    # Divide the two arrays (convert one array to float in order to get float results)
    print("\nDivide a / b:\n{}".format(a / b.astype(float)))


if __name__ == "__main__":
    test_run10()
