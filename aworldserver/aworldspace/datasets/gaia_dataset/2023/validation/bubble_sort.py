def bubble_sort(arr):
    """
    Implements bubble sort algorithm to sort an array in ascending order.
    
    Args:
        arr: The array/list to be sorted
        
    Returns:
        The sorted array
    """
    n = len(arr)
    
    # Traverse through all array elements
    for i in range(n):
        # Flag to optimize if no swaps occur in a pass
        swapped = False
        
        # Last i elements are already in place
        for j in range(0, n-i-1):
            # Traverse the array from 0 to n-i-1
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        
        # If no swapping occurred in this pass, array is sorted
        if not swapped:
            break
    
    return arr

# Example usage
if __name__ == "__main__":
    # Example array
    test_array = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", test_array)
    
    # Sort the array
    sorted_array = bubble_sort(test_array.copy())
    print("Sorted array:", sorted_array)