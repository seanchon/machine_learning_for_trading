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

if __name__ == "__main__":
    test_run()
