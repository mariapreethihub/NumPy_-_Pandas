# NumPy (Numerical Python)

    NumPy is a powerful Python library used for numerical computing. It provides support for large, 
    multi-dimensional arrays and matrices, along with mathematical functions to operate on these arrays 
    efficiently.

  **1.Importing NumPy**
      
      Before using NumPy, you need to import it in your Python script or Jupyter Notebook.

      import numpy as np
      The alias np is commonly used to shorten the name.

  **2. Creating NumPy Arrays**
 
      (A) Creating a 1D NumPy Array
          A NumPy array can be created from a Python list using np.array().   
            
          Example :arr = np.array([1, 2, 3, 4, 5])
                   print(arr)
                   print(type(arr))                               
          Output: <class 'numpy.ndarray'>
      
      (B) Creating a 2D NumPy Array
          A 2D array (or matrix) is created by passing a list of lists.
   
          Example :matrix = np.array([[1, 2, 3], [4, 5, 6]])
                   print(matrix)
          Output:
                  [[1 2 3]
                   [4 5 6]]

      (C) Creating a NumPy Array with Specific Data Type
          We can specify the data type (dtype) while creating an array.
          
          Example:arr_float = np.array([1, 2, 3], dtype=np.float32)
                  print(arr_float)  
                  print(arr_float.dtype)  
          Output: float32

  **3. Creating Special NumPy Arrays**
  
        (A) Creating an Array of Zeros (np.zeros())
            Example : zeros_arr = np.zeros((3, 3))  
                      print(zeros_arr)
            Output  :3x3 array of zeros
            
        (B) Creating an Array of Ones (np.ones())
            Example :ones_arr = np.ones((2, 4))  
                     print(ones_arr)
            Output  : 2x4 array of ones
            
         (C) Creating an Array with a Range (np.arange())
             Example :arr_range = np.arange(1, 10, 2)  # Creates an array starting from 1, go up to 10 (excluding),with step 2
                      print(arr_range)  
              Output: [1 3 5 7 9]
                      
         (D) Creating an Array with Evenly Spaced Values (np.linspace())
             Example :arr_linspace = np.linspace(0, 1, 5)  # 5 equally spaced values between 0 and 1
                      print(arr_linspace)  
             Output: [0.   0.25 0.5  0.75 1.  ]
             
         (E) Creating a Random Array (np.random)
             Example :rand_arr = np.random.rand(3, 3)  
                      print(rand_arr)
             Ouput  :3x3 array with random values between 0 and 1

  **Key Features of NumPy**
  
          > High Performance & Speed – Faster than Python lists due to vectorized operations.
          > Memory Efficiency – Uses less memory compared to Python lists.
          > Broadcasting – Supports element-wise operations without explicit loops.
          > Advanced Indexing & Slicing – Allows slicing, fancy indexing, and Boolean filtering.


# Pandas

      Pandas is a powerful open-source Python library used for data analysis and manipulation. 

  **Types of Data Structures in Pandas**
    
    1. Series – A one-dimensional labeled array (similar to a single column in Excel).
       # Creating a Series from a list
         series = pd.Series(data, index=index)
         
    2. DataFrame – A two-dimensional table (like a spreadsheet).
       # Creating a DataFrame from a dictionary
         df = pd.DataFrame(data, index=index)

**Basic Operations in Pandas**

   **1.Import pandas**
   
       import pandas as pd

   **2.Creating a DataFrame**
   
       data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 35]}
       df = pd.DataFrame(data)

   **3.Viewing Data**
   
       print(df.head())                             # Display first five rows
       print(df.tail())                             # Display last five rows
       
   **4.Adding a Column**
   
     df['Salary'] = [50000, 60000, 55000]

   **5.Filtering Data**
   
     filtered_df = df[df['Age'] > 25]

   **6.Handling Missing Values**
   
     df.fillna(0, inplace=True)                   #Replace Null value with zero
     
   **7.Reading & Writing CSV**
   
      df.to_csv('data.csv', index=False)
      df_new = pd.read_csv('data.csv')
