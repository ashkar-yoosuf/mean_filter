#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define TILE_SIZE 4 

__global__ void meanFilterSharedKernel(float *inputImageKernel, float *outputImagekernel, int imageWidth, int imageHeight)
{
	//Set the row and col value for each thread.
	int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    //initialize shared memory
    __shared__ float sharedmem[(TILE_SIZE+2)][(TILE_SIZE+2)];
    
	//Take some values.
	bool is_x_left = (threadIdx.x == 0), is_x_right = (threadIdx.x == TILE_SIZE-1);
    bool is_y_top = (threadIdx.y == 0), is_y_bottom = (threadIdx.y == TILE_SIZE-1);

	//Initialize with zero
	if(is_x_left)
		sharedmem[threadIdx.x][threadIdx.y+1] = 0;
	else if(is_x_right)
		sharedmem[threadIdx.x + 2][threadIdx.y+1]=0;
	if (is_y_top){
		sharedmem[threadIdx.x+1][threadIdx.y] = 0;
		if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = 0;
		else if(is_x_right)
			sharedmem[threadIdx.x+2][threadIdx.y] = 0;
	}
	else if (is_y_bottom){
		sharedmem[threadIdx.x+1][threadIdx.y+2] = 0;
		if(is_x_right)
			sharedmem[threadIdx.x+2][threadIdx.y+2] = 0;
		else if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y+2] = 0;
	}

	//Setup pixel values
	sharedmem[threadIdx.x+1][threadIdx.y+1] = inputImageKernel[row*imageWidth+col];
	//Check for boundry conditions.
	if(is_x_left && (col>0))
		sharedmem[threadIdx.x][threadIdx.y+1] = inputImageKernel[row*imageWidth+(col-1)];
	else if(is_x_right && (col<imageWidth-1))
		sharedmem[threadIdx.x + 2][threadIdx.y+1]= inputImageKernel[row*imageWidth+(col+1)];
	if (is_y_top && (row>0)){
		sharedmem[threadIdx.x+1][threadIdx.y] = inputImageKernel[(row-1)*imageWidth+col];
		if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y] = inputImageKernel[(row-1)*imageWidth+(col-1)];
		else if(is_x_right )
			sharedmem[threadIdx.x+2][threadIdx.y] = inputImageKernel[(row-1)*imageWidth+(col+1)];
	}
	else if (is_y_bottom && (row<imageHeight-1)){
		sharedmem[threadIdx.x+1][threadIdx.y+2] = inputImageKernel[(row+1)*imageWidth + col];
		if(is_x_right)
			sharedmem[threadIdx.x+2][threadIdx.y+2] = inputImageKernel[(row+1)*imageWidth+(col+1)];
		else if(is_x_left)
			sharedmem[threadIdx.x][threadIdx.y+2] = inputImageKernel[(row+1)*imageWidth+(col-1)];
	}

	__syncthreads();   //Wait for all threads to be done.

	//Setup the filter.
    float filterVector[9] = {sharedmem[threadIdx.x][threadIdx.y], sharedmem[threadIdx.x+1][threadIdx.y],
                    sharedmem[threadIdx.x+2][threadIdx.y], sharedmem[threadIdx.x][threadIdx.y+1],
                    sharedmem[threadIdx.x+1][threadIdx.y+1], sharedmem[threadIdx.x+2][threadIdx.y+1],
                    sharedmem[threadIdx.x] [threadIdx.y+2], sharedmem[threadIdx.x+1][threadIdx.y+2],
                    sharedmem[threadIdx.x+2][threadIdx.y+2]};

	
	{

    float element = 0.0;
    for (int i = 0; i < 9; i++) {
        element += filterVector[i];
    }
	outputImagekernel[row*imageWidth+col] = element/9.0;   //Set the output image values.
	}
}

bool meanFilterGPU(float* image, float* outputImage){
	//Cuda error and image values.
	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaError_t status;
	int width = 1280;
	int height = 1280;

    int size =  width * height * sizeof(float);
    
	//initialize images.
	float *deviceinputimage;
    cudaMalloc((void**) &deviceinputimage, size);
    
	status = cudaGetLastError();              
	if (status != cudaSuccess) {                     
		return false;
    }
    
	cudaMemcpy(deviceinputimage, image, size, cudaMemcpyHostToDevice);
	status = cudaGetLastError();              
	if (status != cudaSuccess) {                     
		cudaFree(deviceinputimage);
		return false;
	}
    
    float *deviceOutputImage;
    cudaMalloc((void**) &deviceOutputImage, size);
    
	//take block and grids.
	dim3 dimBlock(TILE_SIZE, TILE_SIZE);
	dim3 dimGrid((int)ceil((float) width / (float)TILE_SIZE), (int)ceil((float) height / (float)TILE_SIZE));
    
    clock_t start_filter = clock();
    meanFilterSharedKernel<<<dimGrid, dimBlock>>>(deviceinputimage, deviceOutputImage, width, height);
    clock_t end_filter = clock();

    double time_filter = (double)(end_filter-start_filter)/CLOCKS_PER_SEC;
    
    printf("Image size : %d Window size : %d GPU Filter Time: %f\n", width, 3 , time_filter);
	

// save output image to host.
	cudaMemcpy(outputImage, deviceOutputImage, size, cudaMemcpyDeviceToHost);
	status = cudaGetLastError();              
	


if (status != cudaSuccess) {                     
		// std::cout << "Kernel failed for cudaMemcpy cudaMemcpyDeviceToHost: " << cudaGetErrorString(status) << std::endl;
		cudaFree(deviceinputimage);
		cudaFree(deviceOutputImage);
		return false;
	}
	//Free the memory
	cudaFree(deviceinputimage);
	cudaFree(deviceOutputImage);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float time = 0;
	cudaEventElapsedTime(&time,start,stop);
	// printf("time %f\n", time); 
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

	// float tgpushared; //timing variables.
	// clock_t start, end;

	meanFilterGPU(img, filtered_img); //GPU call for median Filtering with shared Kernel.
    
    // for (int i = 0; i < size; i++)
	// 	printf ("%f\n", filtered_img[i]);
}