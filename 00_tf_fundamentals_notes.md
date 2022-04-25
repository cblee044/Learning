
# 02_introduction_to_tensorflow

***Missing earlier parts of section two because I started using Inkdrop in the middle of the section***

**Tensor Attributes**
* Datatype of every element
* Number of dimensions (rank)
* Shape of tensor
  * Number of elements on each axis
* Size of tensor

```python
some_tensor.dtype
some_tensor.ndim
some_tensor.shape
some_tensor.shape[0]
tf.size(some_tensor)
tf.size(some_tensor).numpy() # TURNS THE SIZE OBJECT INTO A NUMPY INTEGER
```

---

**Indexing Tensors**

One can index tensors like Python lists. Let's get the first 2 elements of each dimension of `some_tensor`

```python
some_tensor[:2,:2,:2,:2]
```

Perhaps we want the first element from each dimension from each index except for the final one.

```python
some_tensor[:1,:1,:1,:]
# OR
some_tensor[:1,:1,:1]
```
---
**Reshaping Tensors**

First, make a 2x2 tensor
```python
some_tensor = tf.constant([[10,7],
                           [3,4]])
```

If you want the last iem of each row of this rank 2 tensor

```python
some_tensor[:,-1]
```

Reshaping is performed by adding a new axis. Notice that `...` is used to indicate all other axes. A new axis object is used to add a new axis.

```python
another_tensor = some_tensor[..., tf.newaxis]
```

Another way to accomplish this.

```python
tf.expand_dims(some_tensor, axis=0) # The axis argument is where the next axis will live.
```
---
### **Manipulating Tensors with Tensor Operations**

*Basic operations*
```
+,-,*,/
```
Perform scalar addition with
```python
some_tensor+10
```
This operation will not save on some_tensor unless overwritten

One can perform operations through built-in functions. Using built-in functions is favored because they run faster by utilizing the GPU.
```python
tf.multiply(tensor,10)
```

#### Matrix Multiplication
In machine learning, matrix multiplication(dot product) is one of the most used tensor operations
```python
tf.matmul(some_tensor, another_tensor)
```
Recall from linear algebra that the order matters in dot product. When multiplying a left and right matrix, the number of columns from the left matrix must equal the number of rows of the right matrix.
1. The inner dimensions must match.
2. The shape of the new matrix is equal to the outter dimensions.

This concept is a basic one for the mathematician. If you are one, it is still beneficial to familiarize yourself with the syntax. Let's create two tensors. Can you determine the shape based on the code?
```python
X = tf.constant([[1,2],
                 [3,4],
                 [5,6]])
Y = tf.constant([[7,8],
                 [9,10],
                 [11,12]])
```
Since these tensors break the 1st requirement in matrix multiplication, an error will be raised. 

To multiply these matrices, they can either by reshaped or transposed. Doing so results in left and right matrices that satisfy both requirements.
```python
tf.matmul(X, tf.reshape(Y, shape=(2,3)))
tf.matmul(X, tf.transpose(Y))
```
Note that `reshape()` re-orders matrix Y row-wise into the new shape. 

`tf.tensordot()` is another built-in function that performs the dot product. It requires an additional argument, `axis`.
* `axis=0` performs an interesting operation. With the above shapes, the new matrix has shape (3,2,2,3). I interpret it as the right matrix is scalar multiplied into each element of the left matrix
* `axis=1` performs ordinary matrix multiplication
* `axis=2` performs

#### Changing The Datatype of a Tensor
Changing datatypes of tensors has multiple use cases. The first is to standardize the datatypes in a ML algorithm. Another is to improve the performance of a model by using datatypes with lower memory usage.

From *Mixed Precision* on [TensorFlow](https://www.tensorflow.org/guide/mixed_precision)
> Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. By keeping certain parts of the model in the 32-bit types for numeric stability, the model will have a lower step time and train equally as well in terms of the evaluation metrics such as accuracy. This guide describes how to use the Keras mixed precision API to speed up your models. Using this API can improve performance by more than **3 times** on modern GPUs and **60%** on TPUs.

Consider the following `tf.float32` tensor. This is the default datatype.
```python
B = tf.constant([1.7,2.3])
```
To use a different datatype, specify it as an argument.
```python
C = tf.constant([1.7,2.3], dtype=tf.float16)
```
Of course, you can change the datatype with `cast()`
```python
D = tf.cast(B, dtype=tf.float16)
```
---
### Aggregating Tensor
Aggregating tensors means to condense them from multiple values down to a meaningful smaller value(s). They can also be referred to as statistics of a tensor.
#### Absolute Value
This is pretty straight forward. The function `tf.abs()` will return a same shape matrix with absolute value elements.
```python
D = tf.constant([[-2,-20],
                 [34,1]])
tf.abs(D)
```
#### Mean
The `tf.reduce_mean()` has entire, column-wise, and row-wise mean capabilities.
```python
tf.math.reduce_mean(
    input_tensor, axis=None, keepdims=False, name=None
)

tf.reduce_mean(D) # Computes mean of entire tensor 
tf.reduce_mean(D, axis=0) # Computes column-wise means
tf.reduce_mean(D, axis=1) # Computes row-wise mans
```
---
**01.25.22 [Start]** Today, I will practice how to find min, max and more of tensor. Also, the index at they are.

For context, I will watch lessons on squeezing a tensor, one-hot encoding, and more math operations.

---
#### max()
To avoid writing out bigger tensors I use `tf.random.normal()` to create a random tensor. 

```python
tf.random.normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.dtypes.float32,
    seed=None,
    name=None
)
```
The max function for TensorFlow is `tf.reduce_max()`. When the compared elements are matrices, the maximum in corresponding indices are outputted. For example,
```python
C = tf.random.normal(shape=(2,3,2), mean=1, stddev=1, dtype=tf.float16)
print(C)
tf.reduce_max(C, axis=0)
```
creates a tensor where two matrices of shape 3x2 lie on the zeroth axis. If the max function is called on this axis, the result is a 3x2 matrix that contains the maximumn values at the corresponding.
```
tf.Tensor(
[[[-0.751  -0.1348]
  [ 1.168  -0.3691]
  [-0.676   1.698 ]]

 [[ 0.3872  1.615 ]
  [ 1.543   0.8877]
  [ 0.3755  1.816 ]]], shape=(2, 3, 2), dtype=float16)
<tf.Tensor: shape=(3, 2), dtype=float16, numpy=
array([[0.3872, 1.615 ],
       [1.543 , 0.8877],
       [0.3755, 1.816 ]], dtype=float16)>
```
The original shape is preserved if you include the argument `keepdims`
```python
tf.reduce_max(C, axis=0, keepdims=True)
```
```
<tf.Tensor: shape=(1, 3, 2), dtype=float16, numpy=
array([[[0.3872, 1.615 ],
        [1.543 , 0.8877],
        [0.3755, 1.816 ]]], dtype=float16)>
```
#### min()
Must be similar to above

#### Squeezing a Tensor
Same as before, we make a randomized tensor. This time with a different shape
```python
G = tf.constant(tf.random.normal(shape=(1,1,1,40)))
```
Squeezing a tensor will remove its length-1 dimensions.
```python
G_squeezed = tf.squeeze(G)
```

#### One-Hot Encoding
One-Hot encoding is part of the data-preparation phase that turns categorical variables into dummy variables. Typically, a categorical variable will populate one column. One-Hot encode will 'elongate' the data and create a distinct column for each instance of the categorical variable. Instead of the cell holding the category name or value, it is a '1' or '0'. For example, you have a vector of the number of hours slept over the last week.
```python
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None,
    axis=None,
    dtype=None,
    name=None
)
hours_slept = [6,6,7]
tf.one_hot(hours_slept, depth=7)
```
Rather than a 7x1 vector to represent this data, it is a 7x3 matrix in which a cell with a '1' means that the on the ith day, I slept j hours.
```
<tf.Tensor: shape=(3, 7), dtype=float32, numpy=
array([[0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 1.],
       [0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>
```
Matrices can be encoded as well.
```python
hours_slep = [[6,6,7],
              [6,4,7]]
tf.one_hot(hours_slept, depth=7)
```
The function produces a one-hot encoded matrix for each row. This results in a $(2,3,6)$ shaped tensor. 
```
<tf.Tensor: shape=(2, 3, 7), dtype=float32, numpy=
array([[[0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 0.]],

       [[0., 0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]]], dtype=float32)>
```
> If the input indices is rank N, the output will have rank N+1. The new axis is created at dimension axis (default: the new axis is appended at the end).

#### Tensors and Numpy
Tensors from Tensorflow and arrays from Numpy work well together. Here is an example of creating a tensor from a numpy array and vice versa. I call this nested creation.
```python
J = tf.constant(np.array([2.,4.,5.]))
K = np.array(J)
```
The advantage to this is to draw functionality from both libraries.

Please note that default data types when creating TF tensors and NP arrays(unnested creation through Python lists) are `float32` and `float64` respectively.

---
I was able to complete today's learning goals. Suprisingly, one-hot encoding at the matrix level gave me some trouble. The quote made more sense after I understood the results. Now that I think on it, one-hot encoding 'unravels' the categorical matrix.

---
