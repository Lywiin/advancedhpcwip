#include <stdio.h> 
#include <include/labwork.h> 
#include <cuda_runtime_api.h> 
#include <omp.h>

#define ACTIVE_THREADS 4

int main(int argc, char **argv) {
    printf("USTH ICT Master 2018, Advanced Programming for HPC.\n");
    if (argc < 2) {
        printf("Usage: labwork <lwNum> <inputImage>\n");
        printf("   lwNum        labwork number\n");
        printf("   inputImage   the input file name, in JPEG format\n");
        return 0;
    }

    int lwNum = atoi(argv[1]);
    std::string inputFilename;

    // pre-initialize CUDA to avoid incorrect profiling
    printf("Warming up...\n");
    char *temp;
    cudaMalloc(&temp, 1024);

    Labwork labwork;
    if (lwNum != 2 ) {
        inputFilename = std::string(argv[2]);
        labwork.loadInputImage(inputFilename);
    }

    if (lwNum == 6 ) {
	std::string inputFilename2;
	printf("%s", argv[3]);
	labwork.mode = argv[3];
	printf("%s", labwork.mode);
	if (argc == 5) {
	    inputFilename2 = std::string(argv[4]);
	} else {
	    inputFilename2 = inputFilename;
	}
        labwork.loadInputImage2(inputFilename2);
    }


    printf("Starting labwork %d\n", lwNum);
    Timer timer;
    timer.start();
    switch (lwNum) {
        case 1:
            labwork.labwork1_CPU();
            labwork.saveOutputImage("labwork2-cpu-out.jpg");
            printf("labwork 1 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
            labwork.labwork1_OpenMP();
            labwork.saveOutputImage("labwork2-openmp-out.jpg");
            break;
        case 2:
            labwork.labwork2_GPU();
            break;
        case 3:
	    timer.start();
            labwork.labwork3_GPU();
            labwork.saveOutputImage("labwork3-gpu-out.jpg");
            printf("labwork 3 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 4:
	    timer.start();
            labwork.labwork4_GPU();
            labwork.saveOutputImage("labwork4-gpu-out.jpg");
            printf("labwork 4 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 5:
	    timer.start();
            labwork.labwork5_CPU();
            labwork.saveOutputImage("labwork5-cpu-out.jpg");
            printf("labwork 5 CPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            timer.start();
	    labwork.labwork5_GPU();
            labwork.saveOutputImage("labwork5-gpu-out.jpg");
            printf("labwork 5 GPU ellapsed %.1fms\n", lwNum, timer.getElapsedTimeInMilliSec());
            break;
        case 6:
	    labwork.labwork6_GPU();
            labwork.saveOutputImage("labwork6-gpu-out.jpg");
            break;
        case 7:
            labwork.labwork7_GPU();
            labwork.saveOutputImage("labwork7-gpu-out.jpg");
            break;
        case 8:
            labwork.labwork8_GPU();
            labwork.saveOutputImage("labwork8-gpu-out.jpg");
            break;
        case 9:
            labwork.labwork9_GPU();
            labwork.saveOutputImage("labwork9-gpu-out.jpg");
            break;
        case 10:
            labwork.labwork10_GPU();
            labwork.saveOutputImage("labwork10-gpu-out.jpg");
            break;
    }
}

void Labwork::loadInputImage(std::string inputFileName) {
    inputImage = jpegLoader.load(inputFileName);
}

void Labwork::loadInputImage2(std::string inputFileName2) {
    inputImage2 = jpegLoader.load(inputFileName2);
}

void Labwork::saveOutputImage(std::string outputFileName) {
    jpegLoader.save(outputFileName, outputImage, inputImage->width, inputImage->height, 90);
}

void Labwork::labwork1_CPU() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    for (int j = 0; j < 100; j++) {		// let's do it 100 times, otherwise it's too fast!
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

void Labwork::labwork1_OpenMP() {
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = static_cast<char *>(malloc(pixelCount * 3));
    #pragma omp parallel for
    for (int j = 0; j < 100; j++) {     // let's do it 100 times, otherwise it's too fast!
        #pragma omp parallel for
        for (int i = 0; i < pixelCount; i++) {
            outputImage[i * 3] = (char) (((int) inputImage->buffer[i * 3] + (int) inputImage->buffer[i * 3 + 1] +
                                          (int) inputImage->buffer[i * 3 + 2]) / 3);
            outputImage[i * 3 + 1] = outputImage[i * 3];
            outputImage[i * 3 + 2] = outputImage[i * 3];
        }
    }
}

int getSPcores(cudaDeviceProp devProp) {
    int cores = 0;
    int mp = devProp.multiProcessorCount;
    switch (devProp.major) {
        case 2: // Fermi
            if (devProp.minor == 1) cores = mp * 48;
            else cores = mp * 32;
            break;
        case 3: // Kepler
            cores = mp * 192;
            break;
        case 5: // Maxwell
            cores = mp * 128;
            break;
        case 6: // Pascal
            if (devProp.minor == 1) cores = mp * 128;
            else if (devProp.minor == 0) cores = mp * 64;
            else printf("Unknown device type\n");
            break;
        default:
            printf("Unknown device type\n");
            break;
    }
    return cores;
}

void Labwork::labwork2_GPU() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of GPU %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, i);
	printf("\nDevice number #%d\n", i);
	printf("Device name: %s\n", prop.name);
	printf("Clock rate: %d\n", prop.clockRate);
	printf("Processor count: %d\n", getSPcores(prop));
	printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
	printf("Warp size: %d\n", prop.warpSize); 
    }    
}

__global__ void grayscale(uchar3* input, uchar3* output) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork3_GPU() {
    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch the kernel
	grayscale<<<numBlock, blockSize>>>(devInput, devOutput);

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);

}

__global__ void grayscale2d(uchar3* input, uchar3* output, int width, int height) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;
    output[tid].x = (input[tid].x + input[tid].y + input[tid].z) / 3;
    output[tid].z = output[tid].y = output[tid].x;
}

void Labwork::labwork4_GPU() {
    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	//int blockSize = 1024;
	//int numBlock = pixelCount / blockSize;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3((inputImage->width + 31) / blockSize.x, (inputImage->height + 31) / blockSize.y);

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch the kernel
	grayscale2d<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height);

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
   
}



// CPU implementation of Gaussian Blur
void Labwork::labwork5_CPU() {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };
    int pixelCount = inputImage->width * inputImage->height;
    outputImage = (char*) malloc(pixelCount * sizeof(char) * 3);
    for (int row = 0; row < inputImage->height; row++) {
        for (int col = 0; col < inputImage->width; col++) {
            int sum = 0;
            int c = 0;
            for (int y = -3; y <= 3; y++) {
                for (int x = -3; x <= 3; x++) {
                    int i = col + x;
                    int j = row + y;
                    if (i < 0) continue;
                    if (i >= inputImage->width) continue;
                    if (j < 0) continue;
                    if (j >= inputImage->height) continue;
                    int tid = j * inputImage->width + i;
                    unsigned char gray = (inputImage->buffer[tid * 3] + inputImage->buffer[tid * 3 + 1] + inputImage->buffer[tid * 3 + 2])/3;
                    int coefficient = kernel[(y+3) * 7 + x + 3];
                    sum = sum + gray * coefficient;
                    c += coefficient;
                }
            }
            sum /= c;
            int posOut = row * inputImage->width + col;
            outputImage[posOut * 3] = outputImage[posOut * 3 + 1] = outputImage[posOut * 3 + 2] = sum;
        }
    }
}

__global__ void blur(uchar3* input, uchar3* output, int width, int height) {
    int kernel[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;

    int s = 0;
    int weightsSum = 0;

    for (int row = -3; row <= 3; row++) {
	for (int col = -3; col <= 3; col++) {
	    int tempTid = tid + row * width + col;
	    if (tempTid < 0) continue;
	    if (tempTid >= width * height) continue;

	    int gray = (input[tempTid].x + input[tempTid].y + input[tempTid].z) / 3;
	    s += gray * kernel[(row + 3) * 7 + col + 3];
	    weightsSum += kernel[(row + 3) * 7 + col + 3];
	}
    }

    s /= weightsSum;
    output[tid].x = output[tid].y = output[tid].z = s;
}

__global__ void blurShared(uchar3* input, uchar3* output, int* coefficients, int width, int height) {

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;

    int s = 0;
    int weightsSum = 0;

    int localTid = threadIdx.x + threadIdx.y * blockDim.x;
    __shared__ int scoefficients[49];
    if (localTid < 49) {
	scoefficients[localTid] = coefficients[localTid];
    }
    __syncthreads();    

    for (int row = -3; row <= 3; row++) {
	for (int col = -3; col <= 3; col++) {

	    int tempTidx = tidx + col;
	    int tempTidy = tidy + row;
	    if (tempTidx < 0) return;
	    if (tempTidx >= width) return;
	    if (tempTidy < 0) return;
	    if (tempTidy >= height) return;
	    int tempTid = tempTidx + tempTidy * width;

	    int coef = scoefficients[(row + 3) * 7 + col + 3];

	    int gray = (input[tempTid].x + input[tempTid].y + input[tempTid].z) / 3;
	    s += gray * coef;
	    weightsSum += coef;
	}
    }

    s /= weightsSum;
    output[tid].x = output[tid].y = output[tid].z = s;
}

void Labwork::labwork5_GPU() {
    int coefficients[] = { 0, 0, 1, 2, 1, 0, 0,  
                     0, 3, 13, 22, 13, 3, 0,  
                     1, 13, 59, 97, 59, 13, 1,  
                     2, 22, 97, 159, 97, 22, 2,  
                     1, 13, 59, 97, 59, 13, 1,  
                     0, 3, 13, 22, 13, 3, 0,
                     0, 0, 1, 2, 1, 0, 0 };


    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	//int blockSize = 1024;
	//int numBlock = pixelCount / blockSize;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3((inputImage->width + 31) / blockSize.x, (inputImage->height + 31) / blockSize.y);

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devOutput;
	int *gpuCoefficients;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);
	cudaMalloc(&gpuCoefficients, sizeof(coefficients));

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuCoefficients, coefficients, sizeof(coefficients), cudaMemcpyHostToDevice);

    // launch the kernel
	blurShared<<<gridSize, blockSize>>>(devInput, devOutput, gpuCoefficients, inputImage->width, inputImage->height);

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
	cudaFree(&gpuCoefficients);
    
}

__global__ void binarization(uchar3* input, uchar3* output, int width, int height, int threshold) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;
    
    int gray = (input[tid].x + input[tid].y + input[tid].z) / 3;
    int bPix = (gray / threshold) * 255;

    output[tid].z = output[tid].y = output[tid].x = bPix;
}

__global__ void brightness(uchar3* input, uchar3* output, int width, int height, int brightnessToAdd) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;
    
    int gray = (input[tid].x + input[tid].y + input[tid].z) / 3;
    int bPix = gray + brightnessToAdd;
    
    if (bPix < 0) bPix = 0;
    if (bPix > 255) bPix = 255;

    output[tid].z = output[tid].y = output[tid].x = bPix;
}

__global__ void blending(uchar3* input1, uchar3* input2, uchar3* output, int width, int height, int c) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;
    
    int gray1 = (input1[tid].x + input1[tid].y + input1[tid].z) / 3;
    int gray2 = (input2[tid].x + input2[tid].y + input2[tid].z) / 3;
    int bPix = gray1 * c + gray2 * (1 - c);
    
    output[tid].z = output[tid].y = output[tid].x = bPix;
}

void Labwork::labwork6_GPU() {
    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	//int blockSize = 1024;
	//int numBlock = pixelCount / blockSize;
	dim3 blockSize = dim3(32, 32);
	dim3 gridSize = dim3((inputImage->width + 31) / blockSize.x, (inputImage->height + 31) / blockSize.y);

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devInput2;
	uchar3 *devOutput;
	cudaMalloc(&devInput, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devInput2, inputImage->width * inputImage->height * 3);	
	cudaMalloc(&devOutput, inputImage->width * inputImage->height * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

	char buffer[3];
	int threshold = 128;
	int brightnessToAdd = 10;
	switch(mode[0]) {
	    case 'a':
		printf("Binarization\n");
		printf("Enter a threshold : ");
		scanf("%s", buffer);
		threshold = atoi(buffer);
		// launch binarization kernel
		binarization<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, threshold);
		break;

	    case 'b':
		printf("Brightness\n");
		printf("Enter a brightness value to add : ");
		scanf("%s", buffer);
		brightnessToAdd = atoi(buffer);
		// launch brightness kernel
		brightness<<<gridSize, blockSize>>>(devInput, devOutput, inputImage->width, inputImage->height, brightnessToAdd);
		break;

	    case 'c':
		printf("Blending\n");
		printf("Enter a weight for first image : ", buffer);
		scanf("%s", buffer);
		int c = atoi(buffer);
		
		cudaMemcpy(devInput2, inputImage2->buffer, inputImage2->width * inputImage2->height * 3, cudaMemcpyHostToDevice);
				
		// launch brightness kernel
		blending<<<gridSize, blockSize>>>(devInput, devInput2, devOutput, inputImage->width, inputImage->height, c);
		break;

	}

    // cudaMemcpy: devOutput -> inputImage (host)
	cudaMemcpy(outputImage, devOutput, inputImage->width * inputImage->height * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devInput2);
	cudaFree(&devOutput);

}

/*
//OLD minMax for gray image only
__global__ void minMax(uchar3* input, uchar3* output) {

    extern __shared__ uchar3 cache[];

    int localtid = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * 2 * blockDim.x;

    // precompute the first result, using cache[].x to store the min value and cache[].y to store the max value
    cache[localtid].x = min(input[tid].x, input[tid + blockDim.x].x);
    cache[localtid].y = max(input[tid].y, input[tid + blockDim.x].y);
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
	if (localtid < s) {
	    cache[localtid].x = min(cache[localtid].x, cache[localtid + s].x);
	    cache[localtid].y = max(cache[localtid].y, cache[localtid + s].y);
	}
	__syncthreads();
    }

    // write the result of the block in the first element of the block
    if (localtid == 0) {
	output[blockIdx.x].x = cache[0].x;
	output[blockIdx.x].y = cache[0].y;
    }

}
*/

__global__ void minMax(uchar3* input, uchar3* output) {

    extern __shared__ uchar3 cache[];

    int localtid = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * 2 * blockDim.x;

    // store input values localy temporarly to reduce number of access to global memory
    unsigned char inputX1 = input[tid].x;
    unsigned char inputX2 = input[tid + blockDim.x].x;
    unsigned char inputY1 = input[tid].y;
    unsigned char inputY2 = input[tid + blockDim.x].y;
    unsigned char inputZ1 = input[tid].z;
    unsigned char inputZ2 = input[tid + blockDim.x].z;

    // precompute the first result, using cache[].x to store the min value and cache[].y to store the max value, we use cache[].z too to prevent value of 0 we don't want
    cache[localtid].x = min(min(min(inputX1, inputX2), min(inputY1, inputY2)), min(inputZ1, inputZ2));
    cache[localtid].y = max(max(max(inputX1, inputX2), max(inputY1, inputY2)), max(inputZ1, inputZ2));
    cache[localtid].z = max(max(max(inputX1, inputX2), max(inputY1, inputY2)), max(inputZ1, inputZ2));

    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s /= 2) {
	if (localtid < s) {

	    // update local variables with cache variables
	    inputX1 = cache[localtid].x;
	    inputX2 = cache[localtid + s].x;
	    inputY1 = cache[localtid].y;
	    inputY2 = cache[localtid + s].y;
	    inputZ1 = cache[localtid].z;
	    inputZ2 = cache[localtid + s].z;

	    // compute min and max
	    cache[localtid].x = min(min(min(inputX1, inputX2), min(inputY1, inputY2)), min(inputZ1, inputZ2));
	    cache[localtid].y = max(max(max(inputX1, inputX2), max(inputY1, inputY2)), max(inputZ1, inputZ2));
	    cache[localtid].z = max(max(max(inputX1, inputX2), max(inputY1, inputY2)), max(inputZ1, inputZ2));
	}
	__syncthreads();
    }

    // write the result of the block in the first element of the block
    if (localtid == 0) {
	output[blockIdx.x].x = cache[0].x;
	output[blockIdx.x].y = cache[0].y;
	output[blockIdx.x].z = cache[0].y;
    }
}

__global__ void stretch(uchar3* input, uchar3* output, unsigned int minValue, unsigned int maxValue) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    float f = (float)((unsigned int)input[tid].x - minValue) / (float)(maxValue - minValue);
    int g = f * 255;

    output[tid].z = output[tid].y = output[tid].x = g;
}
	
void Labwork::labwork7_GPU() {

    // init some variables    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	uchar3 *grayImage = static_cast<uchar3 *>(malloc(pixelCount * 3));
	uchar3 *minMaxResult = static_cast<uchar3 *>(malloc(sizeof(char) *3));

	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;

    // cuda malloc
	uchar3 *devInput;
	uchar3 *devOutput;
	uchar3 *devGrayOutput;
	uchar3 *devTempOutput;
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	cudaMalloc(&devGrayOutput, pixelCount * 3);
	cudaMalloc(&devTempOutput, numBlock * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch grayscale kernel to get gray image
	grayscale<<<numBlock, blockSize>>>(devInput, devGrayOutput);

    // launch minMax kernel once to get numBlock number of result that we will reduce after
	minMax<<<numBlock, blockSize / 2, blockSize * sizeof(unsigned char) * 3>>>(devGrayOutput, devTempOutput);

    // launch the same kernel again to reduce the number of result below the size of a block
	int numBlockTemp = numBlock;
	while (numBlockTemp > blockSize) {
	    numBlockTemp /= blockSize;
	    minMax<<<numBlockTemp, blockSize / 2, blockSize * sizeof(char) * 3>>>(devTempOutput, devTempOutput);
	}

    // launch the same kernel with one last block to compute the final result
	minMax<<<1, blockSize / 2, blockSize * sizeof(char) * 3>>>(devTempOutput, devTempOutput);	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

    // cudaMemcpy: devOutput -> minMaxResult (host), get the result as uchar3
	cudaMemcpy(minMaxResult, devTempOutput, sizeof(char) * 3, cudaMemcpyDeviceToHost);

    // get the min value from the x and the max value on the y as computed in the kernel
	unsigned int minValue = minMaxResult[0].x;
	unsigned int maxValue = minMaxResult[0].y;

	printf("MIN VALUE : %d\n", minValue);
	printf("MAX VALUE : %d\n", maxValue);
	
    // launch kernel to recalculate intensity for each pixel
	stretch<<<numBlock, blockSize>>>(devGrayOutput, devOutput, minValue, maxValue);

    // copy the result back to outputImage
	cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
	cudaFree(&devGrayOutput);
	cudaFree(&devTempOutput);
	free(grayImage);
	free(minMaxResult);
}

__global__ void rgb2hsv(uchar3* input, double* outh, double* outs, double* outv, int width, int height) {

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;

    // use local variables to save memory access
    double r = (double)input[tid].x / 255.0;
    double g = (double)input[tid].y / 255.0;
    double b = (double)input[tid].z / 255.0;

    // determine minValue
    double minValue = r < g ? r : g;
    minValue = minValue < b ? minValue : b;

    // determine maxValue
    double maxValue = r > g ? r : g;
    maxValue = maxValue > b ? maxValue : b;

    double delta = maxValue - minValue;

    // use local variables to save memory access
    double h = 0.0;
    double s = 0.0;
    double v = 0.0;

    // set the V
    v = maxValue;

    // set the H
    if (delta == 0) h = 0;
    if (r >= maxValue) 
	h = ((int)((g - b) / delta) % 6) * 60.0;
    else if (g >= maxValue) 
	h = ((b - r) / delta + 2.0) * 60.0;
    else 
	h = ((r - g) / delta + 4.0) * 60.0;

    // set the s
    if (maxValue == 0)
	s = 0;
    else
	s = delta / maxValue;

    // write back result in outputs
    outh[tid] = h;
    outs[tid] = s;
    outv[tid] = v;
}

__global__ void hsv2rgb(double* inh, double* ins, double* inv, uchar3* output, int width, int height) {

    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (tidx > width) return;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy > height) return;
    int tid = tidx + tidy * width;

    // use local variables to save memory access
    double h = inh[tid];
    double s = ins[tid];
    double v = inv[tid];

    // calculate d, hi, f
    double d = h / 60.0;
    int hi = (int)d % 6;
    double f = d - hi;

    // calculate l, m, n
    double l = v * (1 - s);
    double m = v * (1 - f * s);
    double n = v * (1 - (1 - f) * s);

    // use local variables to save memory access
    double fr = 0.0;
    double fg = 0.0;
    double fb = 0.0;

    // switch on the hue using hi that we calculated earlier
    switch(hi) {
	case 0:
	    fr = v; fg = n; fb = l;
	    break;
	case 1:
	    fr = m; fg = v; fb = l;
	    break;
	case 2:
	    fr = l; fg = v; fb = n;
	    break;	 
	case 3:
	    fr = l; fg = m; fb = v;
	    break;
	case 4:
	    fr = n; fg = l; fb = v;
	    break;
	default:
	    fr = v; fg = l; fb = m;
	    break;
    }

    // convert back from [0...1] to [0...255]
    int r = int(fr * 255.0);
    int g = int(fg * 255.0);
    int b = int(fb * 255.0);

    // write back the result in output image
    output[tid].x = r;
    output[tid].y = g;
    output[tid].z = b;

}

void Labwork::labwork8_GPU() {

    // init some variables    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	uchar3 *minMaxResult = static_cast<uchar3 *>(malloc(sizeof(char) *3));

	dim3 block2DSize = dim3(32, 32);
	dim3 gridSize = dim3((inputImage->width + 31) / block2DSize.x, (inputImage->height + 31) / block2DSize.y);

    // cuda malloc
	uchar3 *devInput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);

	double *devOutputH;
	double *devOutputS;
	double *devOutputV;
	cudaMalloc(&devOutputH, pixelCount * sizeof(double));
	cudaMalloc(&devOutputS, pixelCount * sizeof(double));
	cudaMalloc(&devOutputV, pixelCount * sizeof(double));

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch kernel for rgb to hsv convertion
	rgb2hsv<<<gridSize, block2DSize>>>(devInput, devOutputH, devOutputS, devOutputV, inputImage->width, inputImage->height);

    // launch kernel for rgb to hsv convertion
	hsv2rgb<<<gridSize, block2DSize>>>(devOutputH, devOutputS, devOutputV, devOutput, inputImage->width, inputImage->height);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

    // copy the result back to outputImage
	cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);
	
    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
	cudaFree(&devOutputH);
	cudaFree(&devOutputS);
	cudaFree(&devOutputV);
}




typedef struct {
    int values[256];
} Histo;

__global__ void localHisto(uchar3* input, Histo* output, int width, int height, int regionSize) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    Histo lhisto;
    int tempTid = 0;

    for (int i = 0; i < regionSize; i++) {
	tempTid = tid * regionSize + i;
	if (tempTid > width * height)
	    continue;
	lhisto.values[input[tempTid].x] += 1;
    }

    for (int i = 0; i < 256; i++) {
	output[tid].values[i] = lhisto.values[i];
    }
}

__global__ void localHistoGather(Histo* input, int histoCount) {

    int localtid = threadIdx.x;
    int tid = blockIdx.x;

    int halfHistoCount = ceil((double)histoCount / 2);
    if (tid + halfHistoCount >= histoCount) return;
    input[tid].values[localtid] += input[tid + halfHistoCount].values[localtid];
}


__global__ void histoProba(Histo* input, int n, double* proba) {
    
    int localtid = threadIdx.x;

    proba[localtid] = ((double)input[0].values[localtid] / n);
    __syncthreads();
}

__global__ void computeCdf(double* input, int* output) {

    double cumul = 0;

    for (int i = 0; i < 256; i++) {
	cumul += input[i];
	output[i] = (int)(cumul * 255.0);
    }
}

__global__ void equalize(uchar3* input, uchar3* output, int* h) {
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    int e = h[input[tid].x];

    output[tid].x = output[tid].y = output[tid].z = e;
}

void Labwork::labwork9_GPU() {
    // init some variables    
	int pixelCount = inputImage->width * inputImage->height;
	int regionSize = inputImage->width;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));

	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;
	int numBlockRegion = ceil((double)numBlock / regionSize);

	int histoCount = pixelCount / regionSize;
	int histoNumBlock = histoCount;

	Histo *localHistoResult = static_cast<Histo *>(malloc(histoCount * sizeof(Histo)));

    // cuda malloc
	uchar3 *devInput;
	uchar3 *devOutput;
	uchar3 *devGrayOutput;
	Histo *devLocalOutput;
	double *devProbaJ;
	int *devCdf;
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devOutput, pixelCount * 3);
	cudaMalloc(&devGrayOutput, pixelCount * 3);
	cudaMalloc(&devLocalOutput, histoCount * sizeof(Histo));
	cudaMalloc(&devProbaJ, sizeof(double) * 256);
	cudaMalloc(&devCdf, sizeof(int) * 256);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch grayscale kernel to get gray image
	grayscale<<<numBlock, blockSize>>>(devInput, devGrayOutput);

    // launch kernel for local histogram calculation
	localHisto<<<numBlockRegion, blockSize>>>(devGrayOutput, devLocalOutput, inputImage->width, inputImage->height, regionSize);

    // launch kernel to gather histo results
	do {
		histoCount = histoNumBlock;
		histoNumBlock = ceil((double)histoCount / 2);
		localHistoGather<<<histoNumBlock, 256>>>(devLocalOutput, histoCount);
	} while (histoCount > 1);

/*
	cudaMemcpy(localHistoResult, devLocalOutput, histoCount * sizeof(Histo), cudaMemcpyDeviceToHost);

	int s = 0;
	for (int i = 0; i < 256; i++) {
		printf("%d: %d\n", i, localHistoResult[0].values[i]);
		s += localHistoResult[0].values[i];
	}
	printf("SUM: %d\n", s);
	printf("PX COUNT: %d\n", pixelCount);
*/

    // calculate probability of given intensity j
	histoProba<<<1, 256>>>(devLocalOutput, pixelCount, devProbaJ);

/*
	double probaJ[256];
	cudaMemcpy(probaJ, devProbaJ, 256 * sizeof(double), cudaMemcpyDeviceToHost);
	
	double s = 0;
	for (int i = 0; i < 256; i++) {
		printf("%d: %lf\n", i, probaJ[i]); 
		s += probaJ[i];
	}
	printf("SUM: %lf\n", s);
*/

    // calculate cdf array of range [0 ... 255]
	computeCdf<<<1, 1>>>(devProbaJ, devCdf);

/*
	int cdf[256];
	cudaMemcpy(cdf, devCdf, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < 256; i++) {
		printf("%d: %d\n", i, cdf[i]); 
	}
*/

    // equalize image
	equalize<<<numBlock, blockSize>>>(devGrayOutput, devOutput, devCdf);


    // copy the result back to outputImage
	cudaMemcpy(outputImage, devOutput, pixelCount * 3, cudaMemcpyDeviceToHost);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

	
    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devOutput);
	cudaFree(&devGrayOutput);
	cudaFree(&devLocalOutput);
	cudaFree(&devProbaJ);
	cudaFree(&devCdf);
	free(localHistoResult);
}

void Labwork::labwork10_GPU() {

}

