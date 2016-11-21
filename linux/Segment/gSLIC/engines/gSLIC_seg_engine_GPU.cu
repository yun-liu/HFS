#include "gSLIC_seg_engine_GPU.h"
#include "gSLIC_seg_engine_shared.h"

using namespace std;
using namespace gSLIC;
using namespace gSLIC::objects;
using namespace gSLIC::engines;

// ----------------------------------------------------
//
//	kernel function defines
//
// ----------------------------------------------------

__global__ void Cvt_Img_Space_device(const Vector4u* inimg, Vector4f* outimg, Vector2i img_size, COLOR_SPACE color_space);

__global__ void Enforce_Connectivity_device(const int* in_idx_img, int* out_idx_img, Vector2i img_size);

__global__ void Init_Cluster_Centers_device(const Vector4f* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size);

__global__ void Find_Center_Association_device(const Vector4f* inimg, const spixel_info* in_spixel_map, int* out_idx_img, Vector2i map_size, Vector2i img_size, int spixel_size, float weight, float max_xy_dist, float max_color_dist);

__global__ void Update_Cluster_Center_device(const Vector4f* inimg, const int* in_idx_img, spixel_info* accum_map, Vector2i map_size, Vector2i img_size, int spixel_size, int no_blocks_per_line);

__global__ void Finalize_Reduction_Result_device(const spixel_info* accum_map, spixel_info* spixel_list, Vector2i map_size, int no_blocks_per_spixel);

__global__ void Draw_Segmentation_Result_device(const int* idx_img, Vector4u* sourceimg, Vector4u* outimg, Vector2i img_size);

__global__ void Enforce_Connectivity_device1_2(const int* in_idx_img, int* out_idx_img, Vector2i img_size);

// ----------------------------------------------------
//
//	host function implementations
//
// ----------------------------------------------------

seg_engine_GPU::seg_engine_GPU(const settings& in_settings) : seg_engine(in_settings)
{
	source_img = new UChar4Image(in_settings.img_size,true,true);
	cvt_img = new Float4Image(in_settings.img_size, true, true);
	idx_img = new IntImage(in_settings.img_size, true, true);
	tmp_idx_img = new IntImage(in_settings.img_size, true, true);

	if (in_settings.seg_method == DEFAULT_SIZE)
	{
		spixel_size = 8;
	}
	else
	{
		spixel_size = in_settings.spixel_size;
	}

	int spixel_per_col = (int)ceil((float)in_settings.img_size.x / (float)spixel_size);
	int spixel_per_row = (int)ceil((float)in_settings.img_size.y / (float)spixel_size);

	map_size = Vector2i(spixel_per_col, spixel_per_row);
	spixel_map = new SpixelMap(map_size, true, true);

	no_grid_per_center = (int)ceil(spixel_size*3.0f / BLOCK_DIM)*((int)ceil(spixel_size*3.0f / BLOCK_DIM));

	Vector2i accum_size(map_size.x*no_grid_per_center, map_size.y);
	accum_map = new SpixelMap(accum_size, true, true);

	// normalizing factors
	max_xy_dist = 1.0f / (1.4242f * spixel_size); // sqrt(2) * spixel_size
	switch (in_settings.color_space)
	{
	case RGB:
		max_color_dist = 5.0f / (1.7321f * 255);
		break;
	case XYZ:
		max_color_dist = 5.0f / 1.7321f;
		break;
	case CIELAB:
		max_color_dist = 15.0f / (1.7321f * 128);
		break;
	}

	max_color_dist *= max_color_dist;
	max_xy_dist *= max_xy_dist;
}

gSLIC::engines::seg_engine_GPU::~seg_engine_GPU()
{
	delete accum_map;
	delete tmp_idx_img;
}


void gSLIC::engines::seg_engine_GPU::Cvt_Img_Space(UChar4Image* inimg, Float4Image* outimg, COLOR_SPACE color_space)
{
	Vector4u* inimg_ptr = inimg->GetData(MEMORYDEVICE_CUDA);
	Vector4f* outimg_ptr = outimg->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize = getGridSize(img_size, blockSize);
	Cvt_Img_Space_device << <gridSize, blockSize >> >(inimg_ptr, outimg_ptr, img_size, color_space);
}

void gSLIC::engines::seg_engine_GPU::Init_Cluster_Centers()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
	Vector4f* img_ptr = cvt_img->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize = getGridSize(map_size, blockSize);
	Init_Cluster_Centers_device << <gridSize, blockSize >> >(img_ptr, spixel_list, map_size, img_size, spixel_size);
}

void gSLIC::engines::seg_engine_GPU::Find_Center_Association()
{
	spixel_info* spixel_list = spixel_map->GetData(MEMORYDEVICE_CUDA);
	Vector4f* img_ptr = cvt_img->GetData(MEMORYDEVICE_CUDA);
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize = getGridSize(img_size, blockSize);
	Find_Center_Association_device << <gridSize, blockSize >> >(img_ptr, spixel_list, idx_ptr, map_size, img_size, spixel_size, gslic_settings.coh_weight,max_xy_dist,max_color_dist);
}

void gSLIC::engines::seg_engine_GPU::Update_Cluster_Center()
{
	spixel_info* accum_map_ptr = accum_map->GetData(MEMORYDEVICE_CUDA);
	spixel_info* spixel_list_ptr = spixel_map->GetData(MEMORYDEVICE_CUDA);
	Vector4f* img_ptr = cvt_img->GetData(MEMORYDEVICE_CUDA);
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);

	int no_blocks_per_line = (int)ceil(spixel_size * 3.0f / BLOCK_DIM);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize(map_size.x, map_size.y, no_grid_per_center);

	Update_Cluster_Center_device<<<gridSize,blockSize>>>(img_ptr, idx_ptr, accum_map_ptr, map_size, img_size, spixel_size, no_blocks_per_line);

	dim3 gridSize2(map_size.x, map_size.y);

	Finalize_Reduction_Result_device<<<gridSize2,blockSize>>>(accum_map_ptr, spixel_list_ptr, map_size, no_grid_per_center);
}

void gSLIC::engines::seg_engine_GPU::Enforce_Connectivity()
{
	int* idx_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);
	int* tmp_idx_ptr = tmp_idx_img->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize = getGridSize(img_size, blockSize);

	Enforce_Connectivity_device << <gridSize, blockSize >> >(idx_ptr, tmp_idx_ptr, img_size);
	Enforce_Connectivity_device << <gridSize, blockSize >> >(tmp_idx_ptr, idx_ptr, img_size);
	Enforce_Connectivity_device1_2 << <gridSize, blockSize >> >(idx_ptr, tmp_idx_ptr, img_size);
	Enforce_Connectivity_device1_2 << <gridSize, blockSize >> >(tmp_idx_ptr, idx_ptr, img_size);
}

void gSLIC::engines::seg_engine_GPU::Draw_Segmentation_Result(UChar4Image* out_img)
{
	Vector4u* inimg_ptr = source_img->GetData(MEMORYDEVICE_CUDA);
	Vector4u* outimg_ptr = out_img->GetData(MEMORYDEVICE_CUDA);
	int* idx_img_ptr = idx_img->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize = getGridSize(img_size, blockSize);

	Draw_Segmentation_Result_device << <gridSize, blockSize >> >(idx_img_ptr, inimg_ptr, outimg_ptr, img_size);
	out_img->UpdateHostFromDevice();
}

// ----------------------------------------------------
//
//	device function implementations
//
// ----------------------------------------------------

__global__ void Cvt_Img_Space_device(const Vector4u* inimg, Vector4f* outimg, Vector2i img_size, COLOR_SPACE color_space)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= img_size.x || y >= img_size.y) return;

	cvt_img_space_shared(inimg, outimg, img_size, x, y, color_space);
}

__global__ void Draw_Segmentation_Result_device(const int* idx_img, Vector4u* sourceimg, Vector4u* outimg, Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x == 0 || y == 0 || x > img_size.x - 2 || y > img_size.y - 2) return;

	draw_superpixel_boundry_shared(idx_img, sourceimg, outimg, img_size, x, y);
}

__global__ void Init_Cluster_Centers_device(const Vector4f* inimg, spixel_info* out_spixel, Vector2i map_size, Vector2i img_size, int spixel_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= map_size.x || y >= map_size.y) return;

	init_cluster_centers_shared(inimg, out_spixel, map_size, img_size, spixel_size, x, y);
}

__global__ void Find_Center_Association_device(const Vector4f* inimg, const spixel_info* in_spixel_map, int* out_idx_img, Vector2i map_size, Vector2i img_size, int spixel_size, float weight, float max_xy_dist, float max_color_dist)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= img_size.x || y >= img_size.y) return;

	find_center_association_shared(inimg, in_spixel_map, out_idx_img, map_size, img_size, spixel_size, weight, x, y,max_xy_dist,max_color_dist);
}

__global__ void Update_Cluster_Center_device(const Vector4f* inimg, const int* in_idx_img, spixel_info* accum_map, Vector2i map_size, Vector2i img_size, int spixel_size, int no_blocks_per_line)
{
	int local_id = threadIdx.y * blockDim.x + threadIdx.x;

	__shared__ float4_ color_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ float2_ xy_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ volatile int count_shared[BLOCK_DIM*BLOCK_DIM];
	__shared__ bool should_add;

	color_shared[local_id] = float4_(0, 0, 0, 0);
	xy_shared[local_id] = float2_(0, 0);
	count_shared[local_id] = 0;
	should_add = false;
	__syncthreads();

	int no_blocks_per_spixel = gridDim.z;

	int spixel_id = blockIdx.y * map_size.x + blockIdx.x;

	// compute the relative position in the search window
	int block_x = blockIdx.z % no_blocks_per_line;
	int block_y = blockIdx.z / no_blocks_per_line;

	int x_offset = block_x * BLOCK_DIM + threadIdx.x;
	int y_offset = block_y * BLOCK_DIM + threadIdx.y;

	if (x_offset < spixel_size * 3 && y_offset < spixel_size * 3)
	{
		// compute the start of the search window
		int x_start = blockIdx.x * spixel_size - spixel_size;
		int y_start = blockIdx.y * spixel_size - spixel_size;

		int x_img = x_start + x_offset;
		int y_img = y_start + y_offset;

		if (x_img >= 0 && x_img < img_size.x && y_img >= 0 && y_img < img_size.y)
		{
			int img_idx = y_img * img_size.x + x_img;
			if (in_idx_img[img_idx] == spixel_id)
			{
				color_shared[local_id] = float4_(inimg[img_idx].x, inimg[img_idx].y, inimg[img_idx].z, inimg[img_idx].w);
				xy_shared[local_id] = float2_(x_img, y_img);
				count_shared[local_id] = 1;
				should_add = true;
			}
		}
	}
	__syncthreads();

	if (should_add)
	{
		if (local_id < 128)
		{
			color_shared[local_id] += color_shared[local_id + 128];
			xy_shared[local_id] += xy_shared[local_id + 128];
			count_shared[local_id] += count_shared[local_id + 128];
		}
		__syncthreads();

		if (local_id < 64)
		{
			color_shared[local_id] += color_shared[local_id + 64];
			xy_shared[local_id] += xy_shared[local_id + 64];
			count_shared[local_id] += count_shared[local_id + 64];
		}
		__syncthreads();

		if (local_id < 32)
		{
			color_shared[local_id] += color_shared[local_id + 32];
			color_shared[local_id] += color_shared[local_id + 16];
			color_shared[local_id] += color_shared[local_id + 8];
			color_shared[local_id] += color_shared[local_id + 4];
			color_shared[local_id] += color_shared[local_id + 2];
			color_shared[local_id] += color_shared[local_id + 1];

			xy_shared[local_id] += xy_shared[local_id + 32];
			xy_shared[local_id] += xy_shared[local_id + 16];
			xy_shared[local_id] += xy_shared[local_id + 8];
			xy_shared[local_id] += xy_shared[local_id + 4];
			xy_shared[local_id] += xy_shared[local_id + 2];
			xy_shared[local_id] += xy_shared[local_id + 1];

			count_shared[local_id] += count_shared[local_id + 32];
			count_shared[local_id] += count_shared[local_id + 16];
			count_shared[local_id] += count_shared[local_id + 8];
			count_shared[local_id] += count_shared[local_id + 4];
			count_shared[local_id] += count_shared[local_id + 2];
			count_shared[local_id] += count_shared[local_id + 1];
		}
	}
	__syncthreads();

	if (local_id == 0)
	{
		int accum_map_idx = spixel_id * no_blocks_per_spixel + blockIdx.z;
		accum_map[accum_map_idx].center = Vector2f(xy_shared[0].x, xy_shared[0].y);
		accum_map[accum_map_idx].color_info = Vector4f(color_shared[0].x, color_shared[0].y, color_shared[0].z, color_shared[0].w);
		accum_map[accum_map_idx].no_pixels = count_shared[0];
	}
}

__global__ void Finalize_Reduction_Result_device(const spixel_info* accum_map, spixel_info* spixel_list, Vector2i map_size, int no_blocks_per_spixel)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= map_size.x || y >= map_size.y) return;

	finalize_reduction_result_shared(accum_map, spixel_list, map_size, no_blocks_per_spixel, x, y);
}

__global__ void Enforce_Connectivity_device(const int* in_idx_img, int* out_idx_img, Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= img_size.x || y >= img_size.y) return;

	supress_local_lable(in_idx_img, out_idx_img, img_size, x, y);
}

__global__ void Enforce_Connectivity_device1_2(const int* in_idx_img, int* out_idx_img, Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= img_size.x || y >= img_size.y) return;

	supress_local_lable_2(in_idx_img, out_idx_img, img_size, x, y);
}
