import timeit
import numpy as np
from numba import jit, cuda

random101 = np.random.randint(10**1, size = (10**1))
random102 = np.random.randint(10**2, size = (10**2))
random103 = np.random.randint(10**3, size = (10**3))
random104 = np.random.randint(10**4, size = (10**4))
random105 = np.random.randint(10**5, size = (10**5))

fsorted101 = np.arange(0,10**1)
fsorted102 = np.arange(0,10**2)
fsorted103 = np.arange(0,10**3)
fsorted104 = np.arange(0,10**4)
fsorted105 = np.arange(0,10**5)

rsorted101 = np.arange(0,10**1)[::-1]
rsorted102 = np.arange(0,10**2)[::-1]
rsorted103 = np.arange(0,10**3)[::-1]
rsorted104 = np.arange(0,10**4)[::-1]
rsorted105 = np.arange(0,10**5)[::-1]

psorted101 = np.concatenate((np.arange(0,(10**1-10**0)), np.random.randint(10**1, size = 10**0)))
psorted102 = np.concatenate((np.arange(0,(10**2-10**1)), np.random.randint(10**2, size = 10**1)))
psorted103 = np.concatenate((np.arange(0,(10**3-10**2)), np.random.randint(10**3, size = 10**2)))
psorted104 = np.concatenate((np.arange(0,(10**4-10**3)), np.random.randint(10**4, size = 10**3)))
psorted105 = np.concatenate((np.arange(0,(10**5-10**4)), np.random.randint(10**5, size = 10**4)))

@jit(target_backend='cuda')
def bubbleSort(arr):
    n = len(arr)

    # Traverse through all array elements
    for i in range(n):
        swapped = False

        # Last i elements are already in place
        for j in range(0, n-i-1):

            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if (swapped == False):
            break

@jit(target_backend='cuda')
def insertionSort(arr):

    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key

@jit(target_backend='cuda')
def partition(array, low, high):

    # Choose the rightmost element as pivot
    pivot = array[high]

    # Pointer for greater element
    i = low - 1

    # Traverse through all elements
    # compare each element with pivot
    for j in range(low, high):
        if array[j] <= pivot:

            # If element smaller than pivot is found
            # swap it with the greater element pointed by i
            i = i + 1

            # Swapping element at i with element at j
            (array[i], array[j]) = (array[j], array[i])

    # Swap the pivot element with
    # the greater element specified by i
    (array[i + 1], array[high]) = (array[high], array[i + 1])

    # Return the position from where partition is done
    return i + 1

@jit(target_backend='cuda')
def quickSort(array, low, high):
    if low < high:

        # Find pivot element such that
        # element smaller than pivot are on the left
        # element greater than pivot are on the right
        pi = partition(array, low, high)

        # Recursive call on the left of pivot
        quickSort(array, low, pi - 1)

        # Recursive call on the right of pivot
        quickSort(array, pi + 1, high)

bubble_sort_time = timeit.timeit("bubbleSort(random105.copy())", globals=globals(), number=1)
print(f"Bubble Sort Time: {bubble_sort_time:.6f} seconds")
insertion_sort_time = timeit.timeit("insertionSort(random105.copy())", globals=globals(), number=1)
print(f"Insertion Sort Time: {insertion_sort_time:.6f} seconds")
quick_sort_time = timeit.timeit("quickSort(random105.copy(), 0, len(random105.copy()) - 1)", globals=globals(), number=1)
print(f"Quick Sort Time: {quick_sort_time:.6f} seconds")