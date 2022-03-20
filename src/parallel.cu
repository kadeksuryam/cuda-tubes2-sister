%%writefile parallel.cu

// parallel.c

#include <stdio.h>

#include <stdlib.h>

#include <chrono>

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN - 1000

/* 
 * Struct Matrix
 *
 * Matrix representation consists of matrix data 
 * and effective dimensions 
 * */
typedef struct Matrix {
  int mat[NMAX][NMAX]; // Matrix cells
  int row_eff; // Matrix effective row
  int col_eff; // Matrix effective column
}
Matrix;

/* 
 * Procedure init_matrix
 * 
 * Initializing newly allocated matrix
 * Setting all data to 0 and effective dimensions according
 * to nrow and ncol 
 * */
void init_matrix(Matrix * m, int nrow, int ncol) {
  m -> row_eff = nrow;
  m -> col_eff = ncol;

  for (int i = 0; i < m -> row_eff; i++) {
    for (int j = 0; j < m -> col_eff; j++) {
      m -> mat[i][j] = 0;
    }
  }
}

/* 
 * Function input_matrix
 *
 * Returns a matrix with values from stdin input
 * */
Matrix input_matrix(int nrow, int ncol) {
  Matrix input;
  init_matrix( & input, nrow, ncol);

  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      scanf("%d", & input.mat[i][j]);
    }
  }

  return input;
}

/* 
 * Procedure print_matrix
 * 
 * Print matrix data
 * */
void print_matrix(Matrix * m) {
  for (int i = 0; i < m -> row_eff; i++) {
    for (int j = 0; j < m -> col_eff; j++) {
      printf("%d ", m -> mat[i][j]);
    }
    printf("\n");
  }
}

/* 
 * Function get_matrix_datarange
 *
 * Returns the range between maximum and minimum
 * element of a matrix
 * */
__device__ int get_matrix_datarange(Matrix * m) {
  int max = DATAMIN;
  int min = DATAMAX;
  for (int i = 0; i < m -> row_eff; i++) {
    for (int j = 0; j < m -> col_eff; j++) {
      int el = m -> mat[i][j];
      if (el > max) max = el;
      if (el < min) min = el;
    }
  }

  return max - min;
}

/*
 * Function supression_op
 *
 * Returns the sum of intermediate value of special multiplication
 * operation where kernel[0][0] corresponds to target[row][col]
 * */
__device__ int supression_op(Matrix * kernel, Matrix * target, int row, int col) {
  int intermediate_sum = 0;
  for (int i = 0; i < kernel -> row_eff; i++) {
    for (int j = 0; j < kernel -> col_eff; j++) {
      intermediate_sum += kernel -> mat[i][j] * target -> mat[row + i][col + j];
    }
  }

  return intermediate_sum;
}

/*
 * Procedure merge_array
 *
 * Merges two subarrays of n with n[left..mid] and n[mid+1..right]
 * to n itself, with n now ordered ascendingly
 * */
__device__ void merge_array(int * n, int left, int mid, int right) {
  int n_left = mid - left + 1;
  int n_right = right - mid;
  int iter_left = 0, iter_right = 0, iter_merged = left;
  //int arr_left[n_left], arr_right[n_right];
  int * arr_left = new int[n_left];
  int * arr_right = new int[n_right];

  for (int i = 0; i < n_left; i++) {
    arr_left[i] = n[i + left];
  }

  for (int i = 0; i < n_right; i++) {
    arr_right[i] = n[i + mid + 1];
  }

  while (iter_left < n_left && iter_right < n_right) {
    if (arr_left[iter_left] <= arr_right[iter_right]) {
      n[iter_merged] = arr_left[iter_left++];
    } else {
      n[iter_merged] = arr_right[iter_right++];
    }
    iter_merged++;
  }

  while (iter_left < n_left) {
    n[iter_merged++] = arr_left[iter_left++];
  }
  while (iter_right < n_right) {
    n[iter_merged++] = arr_right[iter_right++];
  }

  delete[] arr_left;
  delete[] arr_right;
}

/* 
 * Procedure print_array
 *
 * Prints all elements of array n of size to stdout
 * */
void print_array(int * n, int size) {
  for (int i = 0; i < size; i++) printf("%d ", n[i]);
  printf("\n");
}

void print_mat(Matrix * mat) {
  for (int i = 0; i < mat -> row_eff; i++) {
    for (int j = 0; j < mat -> col_eff; j++) {
      printf("%d ", mat -> mat[i][j]);
    }
    printf("\n");
  }
}

/* 
 * Function get_median
 *
 * Returns median of array n of length
 * */
int get_median(int * n, int length) {
  int mid = length / 2;
  if (length & 1) return n[mid];

  return (n[mid - 1] + n[mid]) / 2;
}

/* 
 * Function get_floored_mean
 *
 * Returns floored mean from an array of integers
 * */
long get_floored_mean(int * n, int length) {
  long sum = 0;
  for (int i = 0; i < length; i++) {
    sum += n[i];
  }

  return sum / length;
}

__global__ void convolution(Matrix * d_kernel, Matrix * d_targets, Matrix * d_results, int blockLen) {
  // Calculate global thread pos

  int currBlock = blockIdx.x;
  int t_row = threadIdx.x;
  int t_col = threadIdx.y;

  // printf("%d\n", d_targets[currBlock].mat[0][0]);

  int rowTarget = d_targets[currBlock].row_eff;
  int colTarget = d_targets[currBlock].col_eff;

  if (t_row < rowTarget && t_col < colTarget) {
    d_results[currBlock].mat[t_row][t_col] = supression_op(d_kernel, & d_targets[currBlock], t_row, t_col);
  }

  for (int i = t_row; i < rowTarget; i += blockLen) {
    for (int j = t_col; j < colTarget; j += blockLen) {
      d_results[currBlock].mat[i][j] = supression_op(d_kernel, & d_targets[currBlock], i, j);
    }
  }

  d_results[currBlock].row_eff = rowTarget - d_kernel -> row_eff + 1;
  d_results[currBlock].col_eff = colTarget - d_kernel -> col_eff + 1;
}

__global__ void datarange(Matrix * d_conv_results, int * d_range_results, int numTargets) {
  int currBlock = blockIdx.x;
  int nThreads = blockDim.x * blockDim.x;
  int currThread = currBlock * nThreads + blockDim.x * threadIdx.y + threadIdx.x;

  if (currThread < numTargets) {
    d_range_results[currThread] = get_matrix_datarange( & d_conv_results[currThread]);
  }
}

__global__ void mergesort(int * arr, int l, int r) {
  if (l < r) {
    int mid = (l + (r - l) / 2);

    cudaStream_t par1, par2;

    // sort the left part
    cudaStreamCreateWithFlags( & par1, cudaStreamNonBlocking);
    mergesort << < 1, 1, 0, par1 >>> (arr, l, mid);
    cudaStreamDestroy(par1);

    // sort the right part
    cudaStreamCreateWithFlags( & par2, cudaStreamNonBlocking);
    mergesort << < 1, 1, 0, par2 >>> (arr, mid + 1, r);
    cudaStreamDestroy(par2);

    cudaDeviceSynchronize();

    merge_array(arr, l, mid, r);
  }
}

// main() driver
int main() {
  using std::chrono::high_resolution_clock;
  using std::chrono::duration_cast;
  using std::chrono::duration;
  using std::chrono::milliseconds;

  int kernel_row, kernel_col, target_row, target_col, num_targets;

  // reads kernel's row and column and initalize kernel matrix from input
  scanf("%d %d", & kernel_row, & kernel_col);
  Matrix kernel = input_matrix(kernel_row, kernel_col);

  // reads number of target matrices and their dimensions.
  // initialize array of matrices and array of data ranges (int)
  scanf("%d %d %d", & num_targets, & target_row, & target_col);
  Matrix * arr_mat = (Matrix * ) malloc(num_targets * sizeof(Matrix));
  Matrix * arr_res = (Matrix * ) malloc(num_targets * sizeof(Matrix));
  int arr_range[num_targets];

  // read each target matrix, compute their convolution matrices, and compute their data ranges
  for (int i = 0; i < num_targets; i++) {
    arr_mat[i] = input_matrix(target_row, target_col);
  }

  auto start_all = high_resolution_clock::now();

  // convolution
  // device memory allocation
  auto start_conv = high_resolution_clock::now();
  Matrix * d_kernel;
  Matrix * d_targets;
  Matrix * d_conv_results;
  cudaMalloc((void ** ) & d_kernel, sizeof(Matrix));
  cudaMalloc((void ** ) & d_targets, num_targets * sizeof(Matrix));
  cudaMalloc((void ** ) & d_conv_results, num_targets * sizeof(Matrix));
  cudaMemcpy(d_kernel, & kernel, sizeof(Matrix), cudaMemcpyHostToDevice);
  cudaMemcpy(d_targets, arr_mat, num_targets * sizeof(Matrix), cudaMemcpyHostToDevice);

  int THREADS_CONV = 16;
  int BLOCKS_CONV = num_targets;

  dim3 block_dim_conv(THREADS_CONV, THREADS_CONV);
  dim3 grid_dim_conv(BLOCKS_CONV);

  convolution << < grid_dim_conv, block_dim_conv >>> (d_kernel, d_targets, d_conv_results, THREADS_CONV);

  cudaDeviceSynchronize();

  cudaMemcpy(arr_res, d_conv_results, num_targets * sizeof(Matrix), cudaMemcpyDeviceToHost);
  auto end_conv = high_resolution_clock::now();

  // Datarange ops
  auto start_dr = high_resolution_clock::now();
  int * d_range_results;
  cudaMalloc((void ** ) & d_range_results, num_targets * sizeof(int));

  int THREADS_DR = 16;
  int BLOCKS_DR = ceil((double) num_targets / (double) THREADS_DR * THREADS_DR);

  dim3 block_dim_dr(THREADS_DR, THREADS_DR);
  dim3 grid_dim_dr(BLOCKS_DR);

  datarange << < grid_dim_dr, block_dim_dr >>> (d_conv_results, d_range_results, num_targets);

  cudaDeviceSynchronize();
  cudaMemcpy(arr_range, d_range_results, num_targets * sizeof(int), cudaMemcpyDeviceToHost);
  auto end_dr = high_resolution_clock::now();

  // sort the data range array
  auto start_sort = high_resolution_clock::now();
  int * gpuDrArr;

  cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 16);

  cudaMalloc((void ** ) & gpuDrArr, num_targets * sizeof(int));
  cudaMemcpy(gpuDrArr, & arr_range, num_targets * sizeof(int), cudaMemcpyHostToDevice);

  mergesort << < 1, 1 >>> (gpuDrArr, 0, num_targets - 1);
  cudaDeviceSynchronize();

  cudaMemcpy(arr_range, gpuDrArr, num_targets * sizeof(int), cudaMemcpyDeviceToHost);
  auto end_sort = high_resolution_clock::now();

  int median = get_median(arr_range, num_targets);
  int floored_mean = get_floored_mean(arr_range, num_targets);

  // print the min, max, median, and floored mean of data range array
  printf("%d\n%d\n%d\n%d\n",
    arr_range[0],
    arr_range[num_targets - 1],
    median,
    floored_mean);
  auto end_all = high_resolution_clock::now();

  duration < double, std::milli > conv_time = end_conv - start_conv;
  duration < double, std::milli > dr_time = end_dr - start_dr;
  duration < double, std::milli > sort_time = end_sort - start_sort;
  duration < double, std::milli > overall_time = end_all - start_all;

  printf("\n==============================\n");
  printf("Convolution Operation time: %fms\n", conv_time.count());
  printf("Datarange Operation time: %fms\n", dr_time.count());
  printf("Sorting Operation time: %fms\n", sort_time.count());
  printf("\n");
  printf("Overall time: %fms\n", overall_time.count());
  printf("==============================\n");
  return 0;
}