// parallel.cu

#include<stdio.h>
#include<stdlib.h>

// to make things easier...NMAX is defined 128 (2^7)
#define NMAX 128

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

__constant__ Matrix gKernel[NMAX][NMAX];

__global__ void convolution_2d(Matrix* matrix, Matrix* result, int nRowKernel, int nColKernel) {
    int gridSize = 16*16; 
    // Calculate global thread pos
    int t_row = blockIdx.y*blockDim.y + threadIdx.y;
    int t_col = blockIdx.x*blockDim.x + threadIdx.x;
    // int gridOffset = (int)(t_row*t_col/16*16)*16*16;

    // t_row %= gridSize;
    // t_col %= gridSize;

    int mat_in_row = matrix.row_eff;
    int mat_in_col = matrix.col_eff;
    
    int tmp = 0;
    for(int i=0;i<nRowKernel;i++) {
        for(int j=0;j<nColKernel;j++) {
            if((t_row%gridSize)*nRowKernel + i < mat_in_row && (t_col%gridSize)*nColKernel + j < mat_in_col) {
                tmp += matrix.mat[(t_row%gridSize)*nRowKernel + i][(t_col%gridSize)*nColKernel + j]*gKernel[i][j];
            }
        }
    }

    result[t_row][t_col] = tmp;
}


/* 
 * Procedure init_matrix
 * 
 * Initializing newly allocated matrix
 * Setting all data to 0 and effective dimensions according
 * to nrow and ncol 
 * */
void init_matrix(Matrix *m, int nrow, int ncol) {
	m->row_eff = nrow;
	m->col_eff = nCol;

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

int main() {
	int kernel_row, kernel_col, target_row, target_col, num_targets;
	
	// reads kernel's row and column and initalize kernel matrix from input
	scanf("%d %d", &kernel_row, &kernel_col);
	Matrix kernel = input_matrix(kernel_row, kernel_col);
	
	// reads number of target matrices and their dimensions.
	// initialize array of matrices and array of data ranges (int)
	scanf("%d %d %d", &num_targets, &target_row, &target_col);
	Matrix* arr_mat = (Matrix*)malloc(num_targets * sizeof(Matrix));
    Matrix* arr_conv_res = (Matrix*)malloc(num_targets*sizeof(Matrix));
	int arr_range[num_targets];
	
	// read each target matrix, compute their convolution matrices, and compute their data ranges
	for (int i = 0; i < num_targets; i++) {
		arr_mat[i] = input_matrix(target_row, target_col);
        arr_conv_res[i] = input_matrix(target_row-kernel_row+1, target_col-kernel_col+1);
        size_t bytes_matrix = sizeof(Matrix); 
        // allocate device memory
        Matrix* d_matrix; Matrix* d_result;
        cudaMalloc(&d_matrix, bytes_matrix);
        cudaMalloc(&d_result, bytes_matrix);

        // copy data from host to device
        cudaMemcpy(d_matrix, arr_mat[i], bytes_matrix, cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(gKernel, kernel, bytes_matrix);

        int THREADS = 16;
        int BLOCKS_X = (target_row+THREADS-1)/THREADS;
        int BLOCKS_Y = (target_col+THREADS-1)/THREADS;

        dim3 block_dim(THREADS, THREADS);
        dim3 grid_dim(BLOCKS_X, BLOCKS_Y);

        convolution_2d<<<grid_dim, block_dim>>>(d_matrix, d_result, target_row, target_col);
	}
}