
#include "Magnitude.h"

#define  MAX_HEIGHT   640
#define  MAX_WIDTH    640

__global__ void Derrivative_x_y_device(const gSLIC::uchar *gray_img, int *delta_x, int *delta_y, int *mag, gSLIC::Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	int idx = y*img_size.x + x;

	if (x == 0)
		delta_x[idx] = gray_img[idx + 1] - gray_img[idx];
	else if (x == img_size.x - 1)
		delta_x[idx] = gray_img[idx] - gray_img[idx - 1];
	else
		delta_x[idx] = gray_img[idx + 1] - gray_img[idx - 1];

	if (y == 0)
		delta_y[idx] = gray_img[idx + img_size.x] - gray_img[idx];
	else if (y == img_size.y - 1)
		delta_y[idx] = gray_img[idx] - gray_img[idx - img_size.x];
	else
		delta_y[idx] = gray_img[idx + img_size.x] - gray_img[idx - img_size.x];

	mag[idx] = (int)(0.5 + sqrt((double)(delta_x[idx] * delta_x[idx] + delta_y[idx] * delta_y[idx])));
}

__global__ void Non_max_supp_device(gSLIC::uchar *nms_mag, int *delta_x, int *delta_y, int *mag, gSLIC::Vector2i img_size)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x, y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x > img_size.x - 1 || y > img_size.y - 1) return;

	int idx = y*img_size.x + x;

	if (x == 0 || x == img_size.x - 1 || y == 0 || y == img_size.y - 1) {
		nms_mag[idx] = 0;
		return;
	}
	
	int m00, gx, gy, z1, z2;
	double mag1, mag2, xprep, yprep;

	m00 = mag[idx];
	if (m00 == 0) {
		nms_mag[idx] = 0;
		return;
	}
	else {
		xprep = -(gx = delta_x[idx]) / ((double)m00);
		yprep = (gy = delta_y[idx]) / ((double)m00);
	}

	if (gx >= 0) {
		if (gy >= 0) {
			if (gx >= gy) {
				z1 = mag[idx - 1];
				z2 = mag[idx - img_size.x - 1];
				mag1 = (m00 - z1)*xprep + (z2 - z1)*yprep;

				z1 = mag[idx + 1];
				z2 = mag[idx + img_size.x + 1];
				mag2 = (m00 - z1)*xprep + (z2 - z1)*yprep;
			}
			else {
				z1 = mag[idx - img_size.x];
				z2 = mag[idx - img_size.x - 1];
				mag1 = (z1 - z2)*xprep + (z1 - m00)*yprep;

				z1 = mag[idx + img_size.x];
				z2 = mag[idx + img_size.x + 1];
				mag2 = (z1 - z2)*xprep + (z1 - m00)*yprep;
			}
		}
		else {
			if (gx >= -gy) {
				z1 = mag[idx - 1];
				z2 = mag[idx + img_size.x - 1];
				mag1 = (m00 - z1)*xprep + (z1 - z2)*yprep;

				z1 = mag[idx + 1];
				z2 = mag[idx - img_size.x + 1];
				mag2 = (m00 - z1)*xprep + (z1 - z2)*yprep;
			}
			else {
				z1 = mag[idx + img_size.x];
				z2 = mag[idx + img_size.x - 1];
				mag1 = (z1 - z2)*xprep + (m00 - z1)*yprep;

				z1 = mag[idx - img_size.x];
				z2 = mag[idx - img_size.x + 1];
				mag2 = (z1 - z2)*xprep + (m00 - z1)*yprep;
			}
		}
	}
	else {
		if (gy >= 0) {
			if (-gx >= gy) {
				z1 = mag[idx + 1];
				z2 = mag[idx - img_size.x + 1];
				mag1 = (z1 - m00)*xprep + (z2 - z1)*yprep;

				z1 = mag[idx - 1];
				z2 = mag[idx + img_size.x - 1];
				mag2 = (z1 - m00)*xprep + (z2 - z1)*yprep;
			}
			else {
				z1 = mag[idx - img_size.x];
				z2 = mag[idx - img_size.x + 1];
				mag1 = (z2 - z1)*xprep + (z1 - m00)*yprep;

				z1 = mag[idx + img_size.x];
				z2 = mag[idx + img_size.x - 1];
				mag2 = (z2 - z1)*xprep + (z1 - m00)*yprep;
			}
		}
		else {
			if (-gx > -gy) {
				z1 = mag[idx + 1];
				z2 = mag[idx + img_size.x + 1];
				mag1 = (z1 - m00)*xprep + (z1 - z2)*yprep;

				z1 = mag[idx - 1];
				z2 = mag[idx - img_size.x - 1];
				mag2 = (z1 - m00)*xprep + (z1 - z2)*yprep;
			}
			else {
				z1 = mag[idx + img_size.x];
				z2 = mag[idx + img_size.x + 1];
				mag1 = (z2 - z1)*xprep + (m00 - z1)*yprep;

				z1 = mag[idx - img_size.x];
				z2 = mag[idx - img_size.x - 1];
				mag2 = (z2 - z1)*xprep + (m00 - z1)*yprep;
			}
		}
	}

	if (mag1 > 0 || mag2 >= 0)
		nms_mag[idx] = 0;
	else
		nms_mag[idx] = (gSLIC::uchar)min(max(m00, 0), 255);
}

Magnitude::Magnitude()
{
	gSLIC::Vector2i size(MAX_WIDTH, MAX_HEIGHT);
	delta_x = new gSLIC::IntImage(size, true, true);
	delta_y = new gSLIC::IntImage(size, true, true);
	mag = new gSLIC::IntImage(size, true, true);
	gray_img = new gSLIC::UCharImage(size, true, true);
	nms_mag = new gSLIC::UCharImage(size, true, true);
	img_size = gSLIC::Vector2i(MAX_WIDTH, MAX_HEIGHT);
}

Magnitude::~Magnitude()
{
	delete delta_x;
	delete delta_y;
	delete mag;
	delete nms_mag;
}

void Magnitude::load_image(const Mat& inimg, gSLIC::UCharImage* outimg)
{
	const int _h = inimg.rows, _w = inimg.cols;
	gSLIC::uchar* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);
	for (int y = 0; y < _h; y++) {
		const uchar *ptr = inimg.ptr<uchar>(y);
		for (int x = 0; x < _w; x++) {
			int idx = x + y * _w;
			outimg_ptr[idx] = ptr[x];
		}
	}
}

void Magnitude::load_image(const gSLIC::UCharImage* inimg, Mat& outimg)
{
	const int _h = outimg.rows, _w = outimg.cols;
	const gSLIC::uchar* inimg_ptr = inimg->GetData(MEMORYDEVICE_CPU);
	for (int y = 0; y < _h; y++) {
		uchar *ptr = outimg.ptr<uchar>(y);
		for (int x = 0; x < _w; x++) {
			int idx = x + y * outimg.cols;
			ptr[x] = inimg_ptr[idx];
		}
	}
}

void Magnitude::derrivative_x_y()
{
	gSLIC::uchar *gray_ptr = gray_img->GetData(MEMORYDEVICE_CUDA);
	int *dx_ptr = delta_x->GetData(MEMORYDEVICE_CUDA);
	int *dy_ptr = delta_y->GetData(MEMORYDEVICE_CUDA);
	int *mag_ptr = mag->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Derrivative_x_y_device << <gridSize, blockSize >> >(gray_ptr, dx_ptr, dy_ptr, mag_ptr, img_size);
}

void Magnitude::non_max_supp()
{
	int *dx_ptr = delta_x->GetData(MEMORYDEVICE_CUDA);
	int *dy_ptr = delta_y->GetData(MEMORYDEVICE_CUDA);
	int *mag_ptr = mag->GetData(MEMORYDEVICE_CUDA);
	gSLIC::uchar *nms_ptr = nms_mag->GetData(MEMORYDEVICE_CUDA);

	dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
	dim3 gridSize((int)ceil((float)img_size.x / (float)blockSize.x), (int)ceil((float)img_size.y / (float)blockSize.y));

	Non_max_supp_device << <gridSize, blockSize >> >(nms_ptr, dx_ptr, dy_ptr, mag_ptr, img_size);
}

void Magnitude::process_img(Mat &bgr3u, Mat &mag1u)
{
	Mat gray, blur1u;
	cvtColor(bgr3u, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur1u, Size(7, 7), 1, 1);

	img_size.x = bgr3u.cols;
	img_size.y = bgr3u.rows;

	load_image(blur1u, gray_img);
	gray_img->UpdateDeviceFromHost();
	derrivative_x_y();
	non_max_supp();
	mag1u.create(bgr3u.rows, bgr3u.cols, CV_8UC1);
	nms_mag->UpdateHostFromDevice();
	load_image(nms_mag, mag1u);
}

