import timeit
import numpy as np
from numba import jit, cuda
import pandas as pd
import sys
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(10**6)

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
def quickSort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quickSort(left) + middle + quickSort(right)

b101r = timeit.timeit("bubbleSort(random101.copy())", globals=globals(), number=10)
i101r = timeit.timeit("insertionSort(random101.copy())", globals=globals(), number=10)
q101r = timeit.timeit("quickSort(random101.copy())", globals=globals(), number=10)

print("Random Sorted Arrays 1 Are Done.")

b102r = timeit.timeit("bubbleSort(random102.copy())", globals=globals(), number=10)
i102r = timeit.timeit("insertionSort(random102.copy())", globals=globals(), number=10)
q102r = timeit.timeit("quickSort(random102.copy())", globals=globals(), number=10)

print("Random Sorted Arrays 2 Are Done.")

b103r = timeit.timeit("bubbleSort(random103.copy())", globals=globals(), number=10)
i103r = timeit.timeit("insertionSort(random103.copy())", globals=globals(), number=10)
q103r = timeit.timeit("quickSort(random103.copy())", globals=globals(), number=10)

print("Random Sorted Arrays 3 Are Done.")

b104r = timeit.timeit("bubbleSort(random104.copy())", globals=globals(), number=10)
i104r = timeit.timeit("insertionSort(random104.copy())", globals=globals(), number=10)
q104r = timeit.timeit("quickSort(random104.copy())", globals=globals(), number=10)

print("Random Sorted Arrays 4 Are Done.")

b105r = timeit.timeit("bubbleSort(random105.copy())", globals=globals(), number=10)
i105r = timeit.timeit("insertionSort(random105.copy())", globals=globals(), number=10)
q105r = timeit.timeit("quickSort(random105.copy())", globals=globals(), number=10)

print("Random Sorted Arrays 5 Are Done.")

random_sorted = np.array([b101r, i101r, q101r, b102r, i102r, q102r, b103r, i103r, q103r, b104r, i104r, q104r, b105r, i105r, q105r])

b101f = timeit.timeit("bubbleSort(fsorted101.copy())", globals=globals(), number=10)
i101f = timeit.timeit("insertionSort(fsorted101.copy())", globals=globals(), number=10)
q101f = timeit.timeit("quickSort(fsorted101.copy())", globals=globals(), number=10)

print("Fully Sorted Arrays 1 Are Done.")

b102f = timeit.timeit("bubbleSort(fsorted102.copy())", globals=globals(), number=10)
i102f = timeit.timeit("insertionSort(fsorted102.copy())", globals=globals(), number=10)
q102f = timeit.timeit("quickSort(fsorted102.copy())", globals=globals(), number=10)

print("Fully Sorted Arrays 2 Are Done.")

b103f = timeit.timeit("bubbleSort(fsorted103.copy())", globals=globals(), number=10)
i103f = timeit.timeit("insertionSort(fsorted103.copy())", globals=globals(), number=10)
q103f = timeit.timeit("quickSort(fsorted103.copy())", globals=globals(), number=10)

print("Fully Sorted Arrays 3 Are Done.")

b104f = timeit.timeit("bubbleSort(fsorted104.copy())", globals=globals(), number=10)
i104f = timeit.timeit("insertionSort(fsorted104.copy())", globals=globals(), number=10)
q104f = timeit.timeit("quickSort(fsorted104.copy())", globals=globals(), number=10)

print("Fully Sorted Arrays 4 Are Done.")

b105f = timeit.timeit("bubbleSort(fsorted105.copy())", globals=globals(), number=10)
i105f = timeit.timeit("insertionSort(fsorted105.copy())", globals=globals(), number=10)
q105f = timeit.timeit("quickSort(fsorted105.copy())", globals=globals(), number=10)

print("Fully Sorted Arrays 5 Are Done.")

fully_sorted = np.array([b101f, i101f, q101f, b102f, i102f, q102f, b103f, i103f, q103f, b104f, i104f, q104r, b105f, i105r, q105f])

b101rs = timeit.timeit("bubbleSort(rsorted101.copy())", globals=globals(), number=10)
i101rs = timeit.timeit("insertionSort(rsorted101.copy())", globals=globals(), number=10)
q101rs = timeit.timeit("quickSort(rsorted101.copy())", globals=globals(), number=10)

print("Reverse Sorted Arrays 1 Are Done.")

b102rs = timeit.timeit("bubbleSort(rsorted102.copy())", globals=globals(), number=10)
i102rs = timeit.timeit("insertionSort(rsorted102.copy())", globals=globals(), number=10)
q102rs = timeit.timeit("quickSort(rsorted102.copy())", globals=globals(), number=10)

print("Reverse Sorted Arrays 2 Are Done.")

b103rs = timeit.timeit("bubbleSort(rsorted103.copy())", globals=globals(), number=10)
i103rs = timeit.timeit("insertionSort(rsorted103.copy())", globals=globals(), number=10)
q103rs = timeit.timeit("quickSort(rsorted103.copy())", globals=globals(), number=10)

print("Reverse Sorted Arrays 3 Are Done.")

b104rs = timeit.timeit("bubbleSort(rsorted104.copy())", globals=globals(), number=10)
i104rs = timeit.timeit("insertionSort(rsorted104.copy())", globals=globals(), number=10)
q104rs = timeit.timeit("quickSort(rsorted104.copy())", globals=globals(), number=10)

print("Reverse Sorted Arrays 4 Are Done.")

b105rs = timeit.timeit("bubbleSort(rsorted105.copy())", globals=globals(), number=10)
i105rs = timeit.timeit("insertionSort(rsorted105.copy())", globals=globals(), number=10)
q105rs = timeit.timeit("quickSort(rsorted105.copy())", globals=globals(), number=10)

print("Reverse Sorted Arrays 5 Are Done.")

reverse_sorted = np.array([b101rs, i101rs, q101rs, b102rs, i102rs, q102rs, b103rs, i103rs, q103rs, b104rs, i104rs, q104rs, b105rs, i105rs, q105rs])

b101p = timeit.timeit("bubbleSort(psorted101.copy())", globals=globals(), number=10)
i101p = timeit.timeit("insertionSort(psorted101.copy())", globals=globals(), number=10)
q101p = timeit.timeit("quickSort(psorted101.copy())", globals=globals(), number=10)

print("Partially Sorted Arrays 1 Are Done.")

b102p = timeit.timeit("bubbleSort(psorted102.copy())", globals=globals(), number=10)
i102p = timeit.timeit("insertionSort(psorted102.copy())", globals=globals(), number=10)
q102p = timeit.timeit("quickSort(psorted102.copy())", globals=globals(), number=10)

print("Partially Sorted Arrays 2 Are Done.")

b103p = timeit.timeit("bubbleSort(psorted103.copy())", globals=globals(), number=10)
i103p = timeit.timeit("insertionSort(psorted103.copy())", globals=globals(), number=10)
q103p = timeit.timeit("quickSort(psorted103.copy())", globals=globals(), number=10)

print("Partially Sorted Arrays 3 Are Done.")

b104p = timeit.timeit("bubbleSort(psorted104.copy())", globals=globals(), number=10)
i104p = timeit.timeit("insertionSort(psorted104.copy())", globals=globals(), number=10)
q104p = timeit.timeit("quickSort(psorted104.copy())", globals=globals(), number=10)

print("Partially Sorted Arrays 4 Are Done.")

b105p = timeit.timeit("bubbleSort(psorted105.copy())", globals=globals(), number=10)
i105p = timeit.timeit("insertionSort(psorted105.copy())", globals=globals(), number=10)
q105p = timeit.timeit("quickSort(psorted105.copy())", globals=globals(), number=10)

print("Partially Sorted Arrays 5 Are Done.")

partially_sorted = np.array([b101p, i101p, q101p, b102p, i102p, q102p, b103p, i103p, q103p, b104p, i104p, q104p, b105p, i105p, q105p])

index_names = ["Random", "Fully Sorted", "Reverse Sorted", "Partially Sorted"]
col_names = ["Bubble 10^1", "Insertion 10^1", "Quick 10^1", "Bubble 10^2", "Insertion 10^2", "Quick 10^2",
               "Bubble 10^3", "Insertion 10^3", "Quick 10^3", "Bubble 10^4", "Insertion 10^4", "Quick 10^4",
               "Bubble 10^5", "Insertion 10^5", "Quick 10^5"]

df = pd.DataFrame(data = np.array([random_sorted, fully_sorted, reverse_sorted, partially_sorted]),  
                  index = index_names,  
                  columns = col_names)
df.to_csv("Sorting_Results.csv")