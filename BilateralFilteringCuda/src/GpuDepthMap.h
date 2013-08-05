#pragma once

enum GpuDepthMapType
{
	DEPTH_MAP_TYPE_ANY,
	DEPTH_MAP_TYPE_FLOAT
};

// A class encapsulating a depth-map, very similar to RgbImage
class GpuDepthMap
{
private:
	float* gpuImage;
	size_t gpuImagePitch;
	int w, h;

	GpuDepthMapType mapType;

public:
	GpuDepthMap() : gpuImage(0), gpuImagePitch(0), w(0), h(0) { }
	~GpuDepthMap() { Destroy(); }

	void Create(GpuDepthMapType type, int width, int height);
	void Destroy();

    float*       Get()       { return gpuImage; }
    const float* Get() const { return gpuImage; }
	GpuDepthMapType GetType() const { return mapType; }

	int GetPitch() const { return gpuImagePitch; }
	int GetWidth() const { return w; }
	int GetHeight() const { return h; }
    int getSizeInBytes() const { return h * gpuImagePitch * sizeof(float); };

	// Copies the contents of the depth map to host memory (which should be of the correct size)
    void CopyDataOut(float *hostData );
    void CopyDataIn ( float* const hostData );

	// This method allows nodes to operate without requiring in-place modification, swapping data
	// and responsibilities with the 'other' depth map
    void SwapData( GpuDepthMap & other );
};
