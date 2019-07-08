#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TILE_SIZE 4 

__global__ void meanFilter(float *deviceinputimage, float *deviceOutputImage, int dim)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ float mem_shared[(TILE_SIZE+2)][(TILE_SIZE+2)];
    
	bool is_x_left = (threadIdx.x == 0), is_x_right = (threadIdx.x == TILE_SIZE-1);
    bool is_y_top = (threadIdx.y == 0), is_y_bottom = (threadIdx.y == TILE_SIZE-1);

	if(is_x_left)
		mem_shared[threadIdx.x][threadIdx.y+1] = 0;
	else if(is_x_right)
		mem_shared[threadIdx.x + 2][threadIdx.y+1]=0;
	if (is_y_top){
		mem_shared[threadIdx.x+1][threadIdx.y] = 0;
		if(is_x_left)
			mem_shared[threadIdx.x][threadIdx.y] = 0;
		else if(is_x_right)
			mem_shared[threadIdx.x+2][threadIdx.y] = 0;
	}
	else if (is_y_bottom){
		mem_shared[threadIdx.x+1][threadIdx.y+2] = 0;
		if(is_x_right)
			mem_shared[threadIdx.x+2][threadIdx.y+2] = 0;
		else if(is_x_left)
			mem_shared[threadIdx.x][threadIdx.y+2] = 0;
	}

	mem_shared[threadIdx.x+1][threadIdx.y+1] = deviceinputimage[row*dim+col];

	if(is_x_left && (col>0))
		mem_shared[threadIdx.x][threadIdx.y+1] = deviceinputimage[row*dim+(col-1)];
	else if(is_x_right && (col<dim-1))
		mem_shared[threadIdx.x + 2][threadIdx.y+1]= deviceinputimage[row*dim+(col+1)];
	if (is_y_top && (row>0)){
		mem_shared[threadIdx.x+1][threadIdx.y] = deviceinputimage[(row-1)*dim+col];
		if(is_x_left)
			mem_shared[threadIdx.x][threadIdx.y] = deviceinputimage[(row-1)*dim+(col-1)];
		else if(is_x_right )
			mem_shared[threadIdx.x+2][threadIdx.y] = deviceinputimage[(row-1)*dim+(col+1)];
	}
	else if (is_y_bottom && (row<dim-1)){
		mem_shared[threadIdx.x+1][threadIdx.y+2] = deviceinputimage[(row+1)*dim + col];
		if(is_x_right)
			mem_shared[threadIdx.x+2][threadIdx.y+2] = deviceinputimage[(row+1)*dim+(col+1)];
		else if(is_x_left)
			mem_shared[threadIdx.x][threadIdx.y+2] = deviceinputimage[(row+1)*dim+(col-1)];
	}

	__syncthreads();

    float filterVector[9] = {mem_shared[threadIdx.x][threadIdx.y], mem_shared[threadIdx.x+1][threadIdx.y],
							mem_shared[threadIdx.x+2][threadIdx.y], mem_shared[threadIdx.x][threadIdx.y+1],
							mem_shared[threadIdx.x+1][threadIdx.y+1], mem_shared[threadIdx.x+2][threadIdx.y+1],
							mem_shared[threadIdx.x] [threadIdx.y+2], mem_shared[threadIdx.x+1][threadIdx.y+2],
							mem_shared[threadIdx.x+2][threadIdx.y+2]};

	
	{

    float element = 0.0;
    for (int i = 0; i < 9; i++) {
        element += filterVector[i];
    }
	deviceOutputImage[row*dim+col] = element/9.0;
	}
}

bool meanFilterGPU(float* image, float* outputImage){
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaError_t status;
	int dim = 1280;

    int size =  dim * dim * sizeof(float);
    
	float *inputimage_D;
    cudaMalloc((void**) &inputimage_D, size);
    
	status = cudaGetLastError();              
	if (status != cudaSuccess) {                     
		return false;
    }
    
	cudaMemcpy(inputimage_D, image, size, cudaMemcpyHostToDevice);
	status = cudaGetLastError();              
	if (status != cudaSuccess) {                     
		cudaFree(inputimage_D);
		return false;
	}
    
    float *outputImage_D;
    cudaMalloc((void**) &outputImage_D, size);
    
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float) dim / (float)TILE_SIZE), (int)ceil((float) dim / (float)TILE_SIZE));
    
    clock_t start_filter = clock();
    meanFilter<<<dimGrid, dimBlock>>>(inputimage_D, outputImage_D, dim);
    clock_t end_filter = clock();

    double time_filter = (double)(end_filter-start_filter)/CLOCKS_PER_SEC;
    
    printf("Image size : %d Window size : %d GPU Filter Time: %f\n", dim, 3 , time_filter);
	
	cudaMemcpy(outputImage, outputImage_D, size, cudaMemcpyDeviceToHost);
	status = cudaGetLastError();              
	


if (status != cudaSuccess) {                     
		cudaFree(inputimage_D);
		cudaFree(outputImage_D);
		return false;
	}
	cudaFree(inputimage_D);
	cudaFree(outputImage_D);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time,start,stop);
	return true;
}

int main()
{
    // freopen("filtered640GPU.txt","w",stdout);
    int size = 1280*1280;
    float *img, *filtered_img;
    img = (float *)malloc(size * sizeof(float));
	filtered_img = (float *)malloc(size * sizeof(float));

    FILE* inp;
    inp = fopen("img1280.txt","r");
    char line[6];
    char *endptr;
    int i = 0;
    while(1){
        char r = (char)fgetc(inp);
        int k = 0;
        while(r!='\n' && !feof(inp)){
            line[k++] = r;
            r = (char)fgetc(inp);
        }
        line[k]=0;
        if(feof(inp)){
            break;
        }
        img[i] = strtof(line, &endptr);
        i++;
    }

	meanFilterGPU(img, filtered_img); //GPU call for median Filtering with shared Kernel.
    
    // for (int i = 0; i < size; i++)
	// 	printf ("%f\n", filtered_img[i]);
}