#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>

float *img_arr;
float *img_arr_padded;
float *filtered_img;

void filterCPU(float *IMG_ARR, float *FILTERED_IMG, int IMG_DIM, int IMG_PADDED_DIM, int PAD, int kernel_dim);

int main(int argc, char **argv){
    
    // freopen("filtered_img.txt", "w", stdout);

    int img_dim = atoi(argv[1]);
    int kernel_dim = atoi(argv[2]);

    int img_size = img_dim*img_dim;
    int img_size_padded = (img_dim+kernel_dim-1)*(img_dim+kernel_dim-1);
	
    img_arr = (float *)malloc(img_size * sizeof(float));
    img_arr_padded = (float *)malloc(img_size_padded * sizeof(float));
	filtered_img = (float *)malloc(img_size * sizeof(float));

    FILE* inp;
    inp = fopen("img640.txt", "r");
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
        img_arr[i] = strtof(line, &endptr);
        i++;
    }

    int img_padded_dim = sqrt(img_size_padded);
    int pad = floor(kernel_dim/2);

    for (int row = 0; row < img_padded_dim; row++) {
        if ((row < pad) || row > img_padded_dim - 1 - pad) {
            for (int col = 0; col < img_padded_dim; col++) {
                img_arr_padded[row*img_padded_dim+col] = 0;
            }
        } else {
            for (int col = 0; col < img_padded_dim; col++) {
                if (col < pad || col > img_padded_dim -1 - pad) {
                    img_arr_padded[row*img_padded_dim+col] = 0;
                }
                else {
                    img_arr_padded[row*img_padded_dim+col] = img_arr[(row-pad)*img_dim+(col-pad)];
                }
            }
        }
    }

    free(img_arr);

    clock_t start_filter = clock();
	filterCPU(img_arr_padded, filtered_img, img_dim, img_padded_dim, pad, kernel_dim);
    clock_t end_filter = clock();

    double time_filter = (double)(end_filter-start_filter)/CLOCKS_PER_SEC;

	// for (int i = 0; i < img_size; i++)
	// 	printf ("%f\n", filtered_img[i]);

    printf("Image size : %d Window size : %d CPU Filter Time: %f\n",img_dim, kernel_dim , time_filter);

    free(filtered_img);
    free(img_arr_padded);

    return 0;
}

void filterCPU(float *IMG_ARR, float *FILTERED_IMG, int IMG_DIM, int IMG_PADDED_DIM, int PAD, int kernel_dim){

    for (int row = PAD; row < (IMG_PADDED_DIM - PAD); row++){
        for (int col = PAD; col < (IMG_PADDED_DIM - PAD); col++){
            float element = 0.0;
            for (int i = row-PAD; i < row+PAD+1; i++){
                for (int j = col-PAD; j < col+PAD+1; j++){
                    element += IMG_ARR[i*IMG_PADDED_DIM+j];
                }
            }
            element /= (kernel_dim*kernel_dim);
            FILTERED_IMG[(row-PAD)*IMG_DIM+(col-PAD)] = element;
        }
    }
}