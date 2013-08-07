// source: http://www.cl.cam.ac.uk/research/rainbow/projects/dcbgrid/DCBGrid-preprint.pdf

#include "GpuImage.h"
#include "helper_cuda.h"
#include "AmCudaHelper.cuh"
#include <iostream>

void GpuImage::Create( GpuImageType type, int width, int height )
{
    if ( type != imageType || width != w || height != h )
    {
        Destroy();

        //LOG_EVENT("Creating image");
        switch ( type )
        {
            case IMAGE_TYPE_XRGB32:
                checkCudaErrors(cudaMallocPitch((void**)&gpuImage, &gpuImagePitch, width * sizeof(unsigned int), height));
                break;

            /*case IMAGE_TYPE_FLOAT2:
                checkCudaErrors(cudaMallocPitch((void**)&gpuImage, &gpuImagePitch, width * sizeof(float2), height));
                break;*/
        }
        w = width;
        h = height;
        imageType = type;
    }
}

void GpuImage::Destroy()
{
    //LOG_EVENT("Destroying image");
    CUDA_FREE( gpuImage );
    w = h = 0;
}

void GpuImage::SwapData(GpuImage & other)
{
    std::swap<unsigned int*>(gpuImage, other.gpuImage);
    std::swap<GpuImageType>(imageType, other.imageType);
    std::swap<size_t>(gpuImagePitch, other.gpuImagePitch);
    std::swap<int>(w, other.w);
    std::swap<int>(h, other.h);
}

void GpuImage::CopyDataIn(unsigned int const* hostData)
{
    checkCudaErrors(cudaMemcpy2D(gpuImage, gpuImagePitch, hostData, w * sizeof(unsigned int),
        w * sizeof(unsigned int), h, cudaMemcpyHostToDevice));
}

#if 0
void GpuImage::AsyncCopyFrom(const GpuImage* const other, const GpuExecutionStream & stream)
{
    // Resize image if needed, then copy in the relevant data
    SizeToMatch(*other);
    checkCudaErrors(cudaMemcpy2DAsync(gpuImage, gpuImagePitch, other->gpuImage, other->gpuImagePitch,
        w, h, cudaMemcpyDeviceToDevice, stream.Get()));
}
#endif

void GpuImage::SizeToMatch(const GpuImage & other)
{
    Create(other.imageType, other.GetWidth(), other.GetHeight());
}

__global__ void TransformImage(unsigned int* imageIn, unsigned int inStride, int width, int height, float* imageOut, unsigned int outStride)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < width && y < height)
    {
        unsigned int input = imageIn[inStride * y + x];
        imageOut[outStride * y + x] = (float)((input & 0xFF) + ((input >> 8) & 0xFF) + ((input >> 16) & 0xFF)) / (255.0f * 3);
    }
}

void GpuImage::CopyDataOut(float* hostData)
{
    // Create device memory for output of the transformation kernel
    float* transImage;
    size_t transImagePitch;
    checkCudaErrors( cudaMallocPitch(&transImage, &transImagePitch, w * sizeof(float), h) );

    dim3 blockDimension(32, 8);
    dim3 gridDimension((w - 1) / blockDimension.x + 1, (h - 1) / blockDimension.y + 1);

    //RECORD_KERNEL_LAUNCH("Transform image kernel", gridDimension, blockDimension);
    TransformImage<<<gridDimension, blockDimension>>>(gpuImage, gpuImagePitch / sizeof(unsigned int), w, h, transImage, transImagePitch / sizeof(float));
    //CHECK_KERNEL_ERROR("Transform image kernel");

    // Copy out the transformed result
    checkCudaErrors(cudaMemcpy2D(hostData, w * sizeof(float), transImage, transImagePitch,
        w * sizeof(float), h, cudaMemcpyDeviceToHost));

    CUDA_FREE(transImage);
}

#pragma region Downsampling

__global__ void DownSampleImage(const unsigned int* const imgIn, const int imgInStride, const int outWidth, const int outHeight, const int dsFactor,
                                unsigned int* const imgOut, const int imgOutStride)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if(x < outWidth && y < outHeight)
        imgOut[imgOutStride * y + x] = imgIn[imgInStride * y * dsFactor + x * dsFactor];
}

void GpuImage::DownsampleFrom(const GpuImage* const other, int downsampleFactor)
{
    Create(other->imageType, other->w / downsampleFactor, other->h / downsampleFactor);

    dim3 blockDimension(32, 8);
    dim3 gridDimension((w - 1) / blockDimension.x + 1, (h - 1) / blockDimension.y + 1);

    //RECORD_KERNEL_LAUNCH("Down-sample image kernel", gridDimension, blockDimension);

    DownSampleImage<<<gridDimension, blockDimension>>>(other->gpuImage, other->gpuImagePitch / sizeof(unsigned int), w, h, downsampleFactor,
        gpuImage, gpuImagePitch / sizeof(unsigned int));

    //CHECK_KERNEL_ERROR("Down-sample image kernel");
}

#pragma endregion
