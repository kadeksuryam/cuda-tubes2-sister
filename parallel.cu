%%writefile parallel.cu

// parallel.c

#include <stdio.h>
#include <stdlib.h>

#define NMAX 100
#define DATAMAX 1000
#define DATAMIN -1000

/* 
 * Struct Matrix
 *
 * Matrix representation consists of matrix data 
 * and effective dimensions 
 * */
typedef struct Matrix {
	int mat[NMAX][NMAX];	// Matrix cells
	int row_eff;			// Matrix effective row
	int col_eff;			// Matrix effective column
} Matrix;


/* 
 * Procedure init_matrix
 * 
 * Initializing newly allocated matrix
 * Setting all data to 0 and effective dimensions according
 * to nrow and ncol 
 * */
void init_matrix(Matrix *m, int nrow, int ncol) {
	m->row_eff = nrow;
	m->col_eff = ncol;

	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			m->mat[i][j] = 0;
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
	init_matrix(&input, nrow, ncol);

	for (int i = 0; i < nrow; i++) {
		for (int j = 0; j < ncol; j++) {
			scanf("%d", &input.mat[i][j]);
		}
	}

	return input;
}


/* 
 * Procedure print_matrix
 * 
 * Print matrix data
 * */
void print_matrix(Matrix *m) {
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			printf("%d ", m->mat[i][j]);
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
int get_matrix_datarange(Matrix *m) {
	int max = DATAMIN;
	int min = DATAMAX;
	for (int i = 0; i < m->row_eff; i++) {
		for (int j = 0; j < m->col_eff; j++) {
			int el = m->mat[i][j];
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
__device__ int supression_op(Matrix *kernel, Matrix *target, int row, int col) {
	int intermediate_sum = 0;
	for (int i = 0; i < kernel->row_eff; i++) {
		for (int j = 0; j < kernel->col_eff; j++) {
			intermediate_sum += kernel->mat[i][j] * target->mat[row + i][col + j];
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
void merge_array(int *n, int left, int mid, int right) {
	int n_left = mid - left + 1;
	int n_right = right - mid;
	int iter_left = 0, iter_right = 0, iter_merged = left;
	int arr_left[n_left], arr_right[n_right];

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

	while (iter_left < n_left)  {
		n[iter_merged++] = arr_left[iter_left++];
	}
	while (iter_right < n_right) {
		n[iter_merged++] = arr_right[iter_right++];
	} 
}


/* 
 * Procedure merge_sort
 *
 * Sorts array n with merge sort algorithm
 * */
void merge_sort(int *n, int left, int right) {
	if (left < right) {
		int mid = left + (right - left) / 2;

		merge_sort(n, left, mid);
		merge_sort(n, mid + 1, right);

		merge_array(n, left, mid, right);
	}	
}
 

/* 
 * Procedure print_array
 *
 * Prints all elements of array n of size to stdout
 * */
void print_array(int *n, int size) {
	for (int i = 0; i < size; i++ ) printf("%d ", n[i]);
	printf("\n");
}


/* 
 * Function get_median
 *
 * Returns median of array n of length
 * */
int get_median(int *n, int length) {
	int mid = length / 2;
	if (length & 1) return n[mid];

	return (n[mid - 1] + n[mid]) / 2;
}


/* 
 * Function get_floored_mean
 *
 * Returns floored mean from an array of integers
 * */
long get_floored_mean(int *n, int length) {
	long sum = 0;
	for (int i = 0; i < length; i++) {
		sum += n[i];
	}

	return sum / length;
}

__global__ void convolution(Matrix* d_kernel, Matrix* d_targets, Matrix* d_results, int num_target) {
    // Calculate global thread pos

    int currBlock = blockIdx.x;
    int t_row = threadIdx.x;
    int t_col = threadIdx.y;
    int blockLen = num_target;

    // printf("%d\n", d_targets[currBlock].mat[0][0]);
    
    int rowTarget = d_targets[currBlock].row_eff;
    int colTarget = d_targets[currBlock].col_eff;

    if(t_row < rowTarget && t_col < colTarget) {
        d_results[currBlock].mat[t_row][t_col] = supression_op(d_kernel, &d_targets[currBlock], t_row, t_col);
    }
    
    for(int i=t_row;i<rowTarget;i+=blockLen) {
        for(int j=t_col;j<colTarget;j+=blockLen) {
            d_results[currBlock].mat[i][j] = supression_op(d_kernel, &d_targets[currBlock], i, j);
        }
    }
}

void print_mat(Matrix* mat) {
    for(int i=0;i<mat->row_eff;i++) {
        for(int j=0;j<mat->col_eff;j++) {
            printf("%d ", mat->mat[i][j]);
        }
        printf("\n");
    }
}

// main() driver
int main() {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	
	// reads kernel's row and column and initalize kernel matrix from input
	scanf("%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);
	
	// reads number of target matrices and their dimensions.
	// initialize array of matrices and array of data ranges (int)
	scanf("%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
    Matrix* arr_res = (Matrix*)malloc(num_targets * sizeof(Matrix));
	int arr_range[num_targets];
	
	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
    // init_matrix(&arr_res[i], target_row-kernel_row+1, target_col-kernel_col+1);
		// arr_mat[i] = convolution(&kernel, &arr_mat[i]);
		// arr_range[i] = get_matrix_datarange(&arr_mat[i]); 
	}

    // device memory allocation
    Matrix *d_kernel; Matrix *d_targets; Matrix *d_results;
    cudaMalloc((void **) &d_kernel, sizeof(Matrix));
    cudaMalloc((void **) &d_targets, num_targets*sizeof(Matrix));
    cudaMalloc((void **) &d_results, num_targets*sizeof(Matrix));
    cudaMemcpy(d_kernel, &kernel, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, arr_mat, num_targets*sizeof(Matrix), cudaMemcpyHostToDevice);

    int THREADS = 16;
    int BLOCKS = num_targets;

    dim3 block_dim(THREADS, THREADS);
    dim3 grid_dim(BLOCKS);

    convolution<<<grid_dim, block_dim>>>(d_kernel, d_targets, d_results, BLOCKS);

    cudaDeviceSynchronize();

    cudaMemcpy(arr_res, d_results, num_targets*sizeof(Matrix), cudaMemcpyDeviceToHost);
		for(int i=0;i<num_targets;i++){
				arr_res[i].row_eff = target_row-kernel_row+1;
				arr_res[i].col_eff = target_col-kernel_col+1;
		}

    for (int i = 0; i < num_targets; i++) {
      // print_mat(&arr_res[i]);
      arr_range[i] = get_matrix_datarange(&arr_res[i]); 
    }
	// sort the data range array
	merge_sort(arr_range, 0, num_targets - 1);
	
	int median = get_median(arr_range, num_targets);	
	int floored_mean = get_floored_mean(arr_range, num_targets); 

	// print the min, max, median, and floored mean of data range array
	printf("%d\n%d\n%d\n%d\n", 
			arr_range[0], 
			arr_range[num_targets - 1], 
			median, 
			floored_mean);

	
	return 0;
}