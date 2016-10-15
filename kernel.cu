/*
 * Author: Tejeswini
 * Purpose: 2D convolution of image on CPU and GPU
 */

#include <iostream>
#include <stdio.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>
#include <string>
#include <cstring>

#define _CRT_SECURE_NO_DEPRECATE

using namespace std;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

#define RGB_COMPONENT_COLOR 255
static int width= 0;
static int height = 0;

/*------------------------------------------------- PPMPixel -----------------
         | Structure- to store image data
         *-------------------------------------------------------------------*/
typedef struct {
     unsigned char red,green,blue;
} PPMPixel;

/*------------------------------------------------- PPMImage -----------------
         | Structure- to store image dimension and data
         *-------------------------------------------------------------------*/

typedef struct {
     int x, y;
     PPMPixel *data;
} PPMImage;

 /*------------------------------------------------- HandleError -----
         |  Function HandleError
         |
         |  Purpose:  This functions checks the status of CUDA APIs 
         |
         |  Parameters:
         |      cudaError_t	 Error Status
		 |		const char *file A file pointer,
		 |		int line where the errror occured
		 |
         |  Returns:	Void
         *-------------------------------------------------------------------*/

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
        exit( EXIT_FAILURE );
    }
}


/*------------------------------------------------- gpuConvolution_x -----
         |  gpuConvolution_x
         |
         |  Purpose:  This function is called by CPU and executed on GPU 
		 |			  This function is to compute the Convolution of image pixels along x axis on GPU
         |
         |  Parameters:
         |      float *kernel Gausian kernel used to blur the image
		 |		int *kernelSizeX Gausian kernel size
		 |		int *width image width
		 |		int *height image hight
		 |		unsigned char* redPixels pointerto buffer storing redPixels
		 |		unsigned char* greenPixels pointer to buffer storing green pixels
		 |		unsigned char* bluePixelOp  pointer to buffer storing convoluted bluepixels
		 |		unsigned char* greenPixelOp pointer to buffer storing convoluted green pixels
		 |		unsigned char* bluePixelOp  pointer to buffer storing  convoluted bluepixels
		 |
		 |
         |  Returns:	Void
         *-------------------------------------------------------------------*/


__global__ void gpuConvolution_x(float *kernel, int *kernelSizeX,  int *width, int *height, unsigned char* redPixels, unsigned char* greenPixels, unsigned char* bluePixels, unsigned char* redPixelsOp, unsigned char* greenPixelsOp, unsigned char* bluePixelsOp) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > (*width) || j > (*height)) return; 

//	for(int j = 0; j < (*height); j++)
//	{
//	for(int i = 0; i < (*width); i++)
//	{
		
				if(i == 0 || i == 1 || i == 0 || i == 1)
				{
				}
				else {
				/*convolute red, green and blue pixels seperately with gausian kernel along x direction*/
				for(int r = 0; r < 5; r++)
				{
					redPixelsOp[(j*(*width)) + i] += (unsigned char)redPixels[((*width)*j)+(i+r-1)]*kernel[r];
					greenPixelsOp[(j*(*width)) + i] += (unsigned char)greenPixels[((*width)*j)+(i+r-1)]*kernel[r];
					bluePixelsOp[(j*(*width))+i] += (unsigned char)bluePixels[((*width)*j)+(i+r-1)]*kernel[r];
				}

		}
	
	/*wait for all threads to complete*/
	__syncthreads();
//	}
//	}

}


/*------------------------------------------------- gpuConvolution_y -----
         |  gpuConvolution_y
         |
         |  Purpose:  This function is called by CPU and executed on GPU 
		 |			  This function is to compute the Convolution of image pixels along y axis on GPU
         |
         |  Parameters:
         |      float *kernel Gausian kernel used to blur the image
		 |		int *kernelSizeX Gausian kernel size
		 |		int *width image width
		 |		int *height image hight
		 |		unsigned char* redPixelsOp pointerto buffer storing redPixels along x axis
		 |		unsigned char* greenPixelsOp pointer to buffer storing green pixels along x axis
		 |		unsigned char* bluePixelOpy  pointer to buffer storing convoluted bluepixels along y axis
		 |		unsigned char* greenPixelOpy pointer to buffer storing convoluted green pixels along y axis
		 |		unsigned char* bluePixelOpy  pointer to buffer storing  convoluted bluepixels along y axis
		 |
		 |
         |  Returns:	Void
         *-------------------------------------------------------------------*/

__global__ void gpuConvolution_y(float *kernel, int *kernelSizeX,  int *width, int *height, unsigned char *redPixelsOp, unsigned char *greenPixelsOp, unsigned char *bluePixelsOp, unsigned char *redPixelsOpy, unsigned char *greenPixelsOpy, unsigned char *bluePixelsOpy, unsigned char *out) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if(i > (*width) || j > (*height)) return; 

//	for(int j = 0; j < (*height); j++)
//	{
//	for(int i = 0; i < (*width); i++)
//	{
		
				if(i == 0 || i == 1 || i == 0 || i == 1)
				{
				}
				else {
					/*convolute red, green and blue pixels seperately with gausian kernel along Y direction*/
					for(int r = 0; r < 5; r++)
						{
							redPixelsOpy[(j*(*width)) + i] += (unsigned char)redPixelsOp[i+(j+r-1)*(*width)]*kernel[r];
							greenPixelsOpy[(j*(*width)) + i] += (unsigned char)greenPixelsOp[i+(j+r-1)*(*width)]*kernel[r];
							bluePixelsOpy[(j*(*width))+i] += (unsigned char)bluePixelsOp[i+(j+r-1)*(*width)]*kernel[r];
						}

			}

		/*wait for all threads to complete*/
		__syncthreads();	


		//}
	//}

}

/*------------------------------------------------- gpuFunction -----
         |  gpuFunction
         |
         |  Purpose:  This function is called by CPU and executed on CPU 
		 |			  This function allocates memory on GPU device and launches GPU kernels
         |
         |  Parameters:
         |      unsigned char *data input image data
		 |		int width width of image file
		 |		int height hight of image file
		 |		float *kernel  Gusian kernel used to blur image on GPU
		 |		int kernelSize Size of Gausian kernel
		 |		unsigned char* out buffer to store convoluted image data
		 |
         |  Returns:	unsigned char* returns pointer to convoluted image
         *-------------------------------------------------------------------*/

unsigned char* gpuFunction(unsigned char *data, int width, int height, float *kernel, int kernelSize, unsigned char* out){

					/*gausian kernel*/
					float *d_kernelForGpu;

					/*pointers to store image data before convolution*/
					unsigned char *d_redpixel;
					unsigned char *d_greenpixel;
					unsigned char *d_bluepixel;
					
					/*pointers to store image data after x convolution*/
					unsigned char *d_redpixelOp;
					unsigned char *d_greenpixelOp;
					unsigned char *d_bluepixelOp;

					/*pointers to store image data after y convolution*/
					unsigned char *d_redpixelOpy;
					unsigned char *d_greenpixelOpy;
					unsigned char *d_bluepixelOpy;

					/*pointer to input image data*/
					unsigned char *d_data;

					/*pointer to convoluted image*/
					unsigned char *d_out;
					
					/*gausian filter size*/
					int *d_kernelSizeX;

					/*image width and hight*/
					int *d_width;
					int *d_height;



		    		dim3 blocks( ((width / 32)+1), ( (height/32)+1));
					dim3 threads(32, 32); 

					HANDLE_ERROR(cudaMalloc((void**)&d_kernelForGpu, kernelSize*sizeof(float)));
					HANDLE_ERROR(cudaMemcpy(d_kernelForGpu, kernel, kernelSize*sizeof(float), cudaMemcpyHostToDevice));

					HANDLE_ERROR(cudaMalloc(&d_data, width*height*3*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMemcpy(d_data, data, 3*width*height*sizeof(unsigned char), cudaMemcpyHostToDevice));

					HANDLE_ERROR(cudaMalloc(&d_out, width*height*3*sizeof(unsigned char)));

					
					HANDLE_ERROR(cudaMalloc((void**)&d_kernelSizeX, sizeof(int)));
					HANDLE_ERROR(cudaMemcpy(d_kernelSizeX, &kernelSize, sizeof(int), cudaMemcpyHostToDevice));

					HANDLE_ERROR(cudaMalloc((void**)&d_width, sizeof(int)));
					HANDLE_ERROR(cudaMemcpy(d_width, &width, sizeof(int), cudaMemcpyHostToDevice));

					HANDLE_ERROR(cudaMalloc((void**)&d_height, sizeof(int)));
					HANDLE_ERROR(cudaMemcpy(d_height, &height, sizeof(int), cudaMemcpyHostToDevice));


					
	
					unsigned char *redPixels = (unsigned char*)malloc((width*height*sizeof(unsigned char)));
					unsigned char *greenPixels = (unsigned char*)malloc((width*height*sizeof(unsigned char)));
					unsigned char *bluePixels = (unsigned char*)malloc((width*height*sizeof(unsigned char)));

					for(int i = 0; i < (width*height);i++ ){
						redPixels[i] = data[i*3+0];
						greenPixels[i] = data[i*3+1];
						bluePixels[i] = data[i*3+2];
					}					
					
					unsigned char *redPixelOp = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
					unsigned char *greenPixelOp = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
					unsigned char *bluePixelOp = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));

					for(int i = 0; i < (width*height); i++){
					redPixels[i] = data[i*3+0];
					greenPixels[i] = data[i*3+1];
					bluePixels[i] = data[i*3+2];
					}

					HANDLE_ERROR(cudaMalloc(&d_redpixel, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMalloc(&d_redpixelOp, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMalloc(&d_redpixelOpy, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMemcpy(d_redpixel, &redPixels, sizeof(int), cudaMemcpyHostToDevice));

					HANDLE_ERROR(cudaMalloc(&d_greenpixel, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMalloc(&d_greenpixelOp, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMalloc(&d_greenpixelOpy, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMemcpy(d_greenpixel, &greenPixels, sizeof(int), cudaMemcpyHostToDevice));

					HANDLE_ERROR(cudaMalloc(&d_bluepixel, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMalloc(&d_bluepixelOp, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMalloc(&d_bluepixelOpy, width*height*sizeof(unsigned char)));
					HANDLE_ERROR(cudaMemcpy(d_bluepixel, &bluePixels, sizeof(int), cudaMemcpyHostToDevice));

	
					unsigned char *redPixelOpy = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
					unsigned char *greenPixelOpy = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
					unsigned char *bluePixelOpy = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));

					clock_t start = clock();
					
					/*Launch GPU kernels here*/
					gpuConvolution_x<<<blocks, threads>>>(d_kernelForGpu, d_kernelSizeX, d_width, d_height, d_redpixel, d_greenpixel, d_bluepixel, d_redpixelOp, d_greenpixelOp, d_bluepixelOp);

					gpuConvolution_y<<<blocks, threads>>>(d_kernelForGpu, d_kernelSizeX, d_width, d_height, d_redpixelOp, d_greenpixelOp, d_bluepixelOp, d_redpixelOpy, d_greenpixelOpy, d_bluepixelOpy, d_out);

					clock_t end = clock();

					/*profile convolution time here*/
					double elapsed_time = (end - start)/(double)CLOCKS_PER_SEC;
					printf("Convolution time on GPU is %lf fractional seconds\n", elapsed_time);
						

					HANDLE_ERROR(cudaMemcpy(redPixels, d_redpixelOp, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost));
					 HANDLE_ERROR(cudaMemcpy(greenPixels, d_greenpixelOp, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost));
					 HANDLE_ERROR(cudaMemcpy(bluePixels, d_bluepixelOp, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost));

					for(int k = 0; k < (width*height); k++) {
				                out[k*3+0]= redPixelOpy[k];
				                out[k*3+1]= greenPixelOpy[k];
				                out[k*3+2]= bluePixelOpy[k];
        				}

					/*File to store GPU convoluted image*/
					FILE *fp;
					fp = fopen("gpu_out.ppm", "wb");
					if(!fp) {
						exit(1);
					}
					fprintf(fp, "P6\n");
					fprintf(fp, "%d %d\n", width, height);
					fprintf(fp, "%d\n", RGB_COMPONENT_COLOR);
					fwrite(out, 3*width, height, fp);
					fclose(fp);
					
					cudaFree(kernel);
					cudaFree(d_kernelSizeX);
					cudaFree(d_data);
					cudaFree(d_out);
					cudaFree(d_redpixelOpy);
					cudaFree(d_greenpixelOpy);
					cudaFree(d_bluepixelOpy);
					cudaFree(d_redpixelOp);
					cudaFree(d_greenpixelOp);
					cudaFree(d_bluepixelOp);
					cudaFree(d_redpixel);
					cudaFree(d_greenpixel);
					cudaFree(d_bluepixel);
					return out;
}

/*------------------------------------------------- readPPM -----
         |  readPPM
         |
         |  Purpose:  This function is called by CPU and executed on CPU 
		 |			  This function reads the input image passed through 
		 |			  command line argument and stores the data in a buffer
         |
         |  Parameters: constant char* input image file passed through command line argument
         |      
         |  Returns:	unsigned char* returns pointer to input image
         *-------------------------------------------------------------------*/


static unsigned char *readPPM(const char *filename)
{
		 char buff[16];
         PPMImage *img;
         FILE *fp;
		 unsigned char *data;
		 int c, rgb_comp_color;
         

		 fp = fopen(filename, "rb");
        
		 if (!fp) {
              fprintf(stderr, "Unable to open file '%s'\n", filename);
              exit(1);
         }

		 if (!fgets(buff, sizeof(buff), fp)) {
              perror(filename);
              exit(1);
         }

		 if (buff[0] != 'P' || buff[1] != '6') {
			fprintf(stderr, "Invalid image format (must be 'P6')\n");
			exit(1);
		}

	    img = (PPMImage *)malloc(sizeof(PPMImage));
		if (!img) {
         fprintf(stderr, "Unable to allocate memory\n");
         exit(1);
		}

	    c = getc(fp);
		while (c == '#') {
			while (getc(fp) != '\n') ;
				c = getc(fp);
			}

		ungetc(c, fp);
			
		if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
			 fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
			 exit(1);
		 }

		width = img->x;
		height = img->y;

    
		if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
			 fprintf(stderr, "Invalid rgb component (error loading '%s')\n", filename);
			 exit(1);
		}

		if (rgb_comp_color!= RGB_COMPONENT_COLOR) {
			 fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
			 exit(1);
		}

		while (fgetc(fp) != '\n') ;
    
		data = (unsigned char*)malloc(img->x * img->y * 3*sizeof(unsigned char));
		if (!data) {
			 fprintf(stderr, "Unable to allocate memory\n");
			 exit(1);
		}

		if (fread(data, 3 * img->x, img->y, fp));
		fclose(fp);
	
		
		return data;
	
}


/*------------------------------------------------- convolvePPM -----
         |  readPPM
         |
         |  Purpose:  This function is called by CPU and executed on CPU 
		 |			  This function does the convolution of input image 
		 |			  command line argument and stores the data in a buffer
         |
         |  Parameters: unsigned char *img: input image data
		 |				unsigned char *out: convoluted image data 
		 |				float kernel[]: gausian kernel for convolution
		 |				int kernelSizeX: size of gausian kernel
		 |				int kernelSizeY: size of gausian kernel
         |      
         |  Returns:	unsigned char* returns pointer to convoluted image
         *-------------------------------------------------------------------*/



unsigned char *convolvePPM(unsigned char *img, unsigned char *out, float kernel[], int kernelSizeX, int kernelSizeY)
{
		FILE *fp;
		
	
	unsigned char *redPixels = (unsigned char*)malloc((width*height*sizeof(unsigned char)));
	unsigned char *greenPixels = (unsigned char*)malloc((width*height*sizeof(unsigned char)));
	unsigned char *bluePixels = (unsigned char*)malloc((width*height*sizeof(unsigned char)));

	unsigned char *redPixelOp = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
	unsigned char *greenPixelOp = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
	unsigned char *bluePixelOp = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));

	
	unsigned char *redPixelOpy = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
	unsigned char *greenPixelOpy = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
	unsigned char *bluePixelOpy = (unsigned char*)malloc((width*height*3*sizeof(unsigned char)));
	
	for(int i = 0; i < (width*height);i++ ){

		redPixels[i] = img[i*3+0];
		greenPixels[i] = img[i*3+1];
		bluePixels[i] = img[i*3+2];
	}

	
    fp = fopen("CPU_out.ppm", "wb");
    if (!fp) {
         
         exit(1);
    }

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n",width, height);

    fprintf(fp, "%d\n",RGB_COMPONENT_COLOR);

	out[0]=0;

	clock_t start = clock();
	
	for(int q = 0; q < height; q++)
	{
	for(int p = 0; p < width; p++)
	{
		
				if(p == 0 || p == 1 || q == 0 || q == 1)
				{
				}
				else {
				for(int r = 0; r < 5; r++)
				{
					redPixelOp[(q*width) + p] += (unsigned char)redPixels[(width*q)+(p+r-1)]*kernel[r];
					greenPixelOp[(q*width) + p] += (unsigned char)greenPixels[(width*q)+(p+r-1)]*kernel[r];
					bluePixelOp[(q*width)+p] += (unsigned char)bluePixels[(width*q)+(p+r-1)]*kernel[r];
				}

				for(int r = 0; r < 5; r++)
				{
					redPixelOpy[(q*width) + p] += (unsigned char)redPixelOp[p+(q+r-1)*width]*kernel[r];
					greenPixelOpy[(q*width) + p] += (unsigned char)greenPixelOp[p+(q+r-1)*width]*kernel[r];
					bluePixelOpy[(q*width)+p] += (unsigned char)bluePixelOp[p+(q+r-1)*width]*kernel[r];
				}
				}
	}
	}

	clock_t end = clock();
	double elapsed_time = (end - start)/(double)CLOCKS_PER_SEC;

	printf("Convolution time for CPU is %lf fractional seconds\n", elapsed_time);

	for(int k = 0; k < (width*height); k++) {
		out[k*3+0]= redPixelOpy[k];
		out[k*3+1]= greenPixelOpy[k];
		out[k*3+2]= bluePixelOpy[k];
	}

	 fwrite(out, 3 * width, height, fp);

    fclose(fp);

	return out;
}

/*------------------------------------------------- main -----
         |  main
         |
         |  Purpose:  This function is called by CPU and executed on CPU 
		 |			  This function erceives command line arguments, image file 
		 |			  name and the standard deviation
		 |			  Prepares gausian kernel and calls the convolution functions on CPU and GPU
         |
         |  Parameters: int argc: number of commandline arguments
		 |				argv[] : image file name and standard deviation
         |  Returns:	int 0 on success
         *-------------------------------------------------------------------*/

int main(int argc, char* argv[]) {	
	string filename = argv[1];
	unsigned char *image;
	unsigned char *outimageongpu;
	int sigma = atoi(argv[2]);	
	image = readPPM(filename.c_str());
	unsigned char *out = (unsigned char*)malloc(width * height * 3 * sizeof(unsigned char));
	float *gKernel = (float*)malloc(5 * sizeof(float));

	double s = 2.0 * sigma * sigma;
	double sum = 0.0;

	int p = 0;
	for(int x = -2; x <=2; x++)
   {
			gKernel[p] = (exp(-(x*x)/s))/(sqrt(M_PI * s));
			sum += gKernel[p];
			p++;                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    }

	
    for(int i = 0; i < 5; i++) {
            gKernel[i] /= sum;
	
	}

	unsigned char *outimage = convolvePPM(image, out, gKernel, 5, 5);
	outimageongpu = (unsigned char*)malloc(width*height*3*sizeof(unsigned char));
	unsigned char *gpu_outimage = gpuFunction(image, width, height, gKernel, 5, outimageongpu);
}


