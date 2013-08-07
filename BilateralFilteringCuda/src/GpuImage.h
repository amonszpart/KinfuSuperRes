#ifndef __GPU_IMAGE_H
#define __GPU_IMAGE_H
#pragma once

//#include "GpuExecutionStream.h"
#include <stddef.h>

enum GpuImageType
{
	IMAGE_TYPE_ANY,
    IMAGE_TYPE_XRGB32
};

// A very simple, very open, cuda-held image class
class GpuImage
{
private:
	unsigned int* gpuImage;
	size_t gpuImagePitch;
	int w, h;

	GpuImageType imageType;

public:
	GpuImage() : gpuImage(0), gpuImagePitch(0), w(0), h(0) { }
	~GpuImage() { Destroy(); }

	void Create(GpuImageType type, int width, int height);
	void Destroy();

	unsigned int* Get() const { return gpuImage; }
	GpuImageType GetType() const { return imageType; }

	int GetPitch() const { return gpuImagePitch; }
	int GetWidth() const { return w; }
	int GetHeight() const { return h; }

	// This method allows nodes to operate without requiring in-place modification, swapping the resultant data
	// and responsibility to the output grid
	void SwapData(GpuImage & other);

	// Copies from the other to this image, on the specified execution stream
    //void AsyncCopyFrom(const GpuImage* const other, const GpuExecutionStream & stream);
	void SizeToMatch(const GpuImage & other);

	// Copies xrgb data into the image, assuming hostData is dimensioned correctly
    void CopyDataIn(unsigned int const* hostData);

	// Copies the image out to a greyscale floating-point array
	void CopyDataOut(float* hostData);

	// Sets this image to a down-sampled version of 'other'
	void DownsampleFrom(const GpuImage* const other, int downsampleFactor);
};

#endif // __GPU_IMAGE_H
