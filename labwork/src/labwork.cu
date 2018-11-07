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

__global__ void stretchMinMax(uchar3* input, uchar3* output, int width, int height) {

    extern __shared__ uchar3 cache[];

    int localtid = threadIdx.x;
    int tid = threadIdx.x + blockIdx.x * 2 * blockDim.x;

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

    if (localtid == 0) {
	output[blockIdx.x].x = cache[0].x;
	output[blockIdx.x].y = cache[0].y;
    }

}

void Labwork::labwork7_GPU() {

    // inputImage->width, inputImage->height    
	int pixelCount = inputImage->width * inputImage->height;
	outputImage = static_cast<char *>(malloc(pixelCount * 3));
	uchar3 *grayImage = static_cast<uchar3 *>(malloc(pixelCount * 3));
	uchar3 *minMaxResult = static_cast<uchar3 *>(malloc(sizeof(char) *3));

	int blockSize = 1024;
	int numBlock = pixelCount / blockSize;

    // cuda malloc: devInput, devOutput
	uchar3 *devInput;
	uchar3 *devGrayOutput;
	uchar3 *devOutput;
	cudaMalloc(&devInput, pixelCount * 3);
	cudaMalloc(&devGrayOutput, pixelCount * 3);
	cudaMalloc(&devOutput, numBlock * 3);

    // cudaMemcpy: inputImage (hostInput) -> devInput
	cudaMemcpy(devInput, inputImage->buffer, inputImage->width * inputImage->height * 3, cudaMemcpyHostToDevice);

    // launch stretch kernel
	grayscale<<<numBlock, blockSize>>>(devInput, devGrayOutput);
/*
	cudaMemcpy(grayImage, devGrayOutput, pixelCount * 3, cudaMemcpyDeviceToHost);
	for (int i = 0; i < 1000; i+=1) {
	    printf("GRAY IMAGE %d : %d\n", i, grayImage[i].x);
	}
*/

	stretchMinMax<<<numBlock, blockSize / 2, blockSize * sizeof(unsigned char) * 3>>>(devGrayOutput, devOutput, inputImage->width, inputImage->height);

	int numBlockTemp = numBlock;

	while (numBlockTemp > blockSize) {
	    numBlockTemp /= blockSize;
	    stretchMinMax<<<numBlockTemp, blockSize / 2, blockSize * sizeof(char) * 3>>>(devOutput, devOutput, inputImage->width, inputImage->height);
	}

	stretchMinMax<<<1, blockSize / 2, blockSize * sizeof(char) * 3>>>(devOutput, devOutput, inputImage->width, inputImage->height);	

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess)
	    printf("Error: %s\n", cudaGetErrorString(err));

    // cudaMemcpy: devOutput -> minArray (host)
	cudaMemcpy(minMaxResult, devOutput, sizeof(char) * 3, cudaMemcpyDeviceToHost);

	unsigned char minValue = minMaxResult[0].x;
	unsigned char maxValue = minMaxResult[0].y;

	printf("MIN VALUE : %d\n", minValue);
	printf("MAX VALUE : %d\n", maxValue);
	
    // cudaFree
	cudaFree(&devInput);
	cudaFree(&devGrayOutput);
	cudaFree(&devOutput);
	free(grayImage);
	free(minMaxResult);
}

void Labwork::labwork8_GPU() {

}

void Labwork::labwork9_GPU() {

}

void Labwork::labwork10_GPU() {

}


