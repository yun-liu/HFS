
#include "Texture.h"
#include "Magnitude.h"

unsigned Texture::support_x_rotated(unsigned support_x, unsigned support_y, double ori)
{
	const double sx_cos_ori = (double)support_x * cos(ori);
	const double sy_sin_ori = (double)support_y * sin(ori);
	double x0_mag = abs(sx_cos_ori - sy_sin_ori);
	double x1_mag = abs(sx_cos_ori + sy_sin_ori);
	return (unsigned)((x0_mag > x1_mag) ? x0_mag : x1_mag);
}

unsigned Texture::support_y_rotated(unsigned support_x, unsigned support_y, double ori)
{
	const double sx_sin_ori = (double)support_x * sin(ori);
	const double sy_cos_ori = (double)support_y * cos(ori);
	double y0_mag = abs(sx_sin_ori - sy_cos_ori);
	double y1_mag = abs(sx_sin_ori + sy_cos_ori);
	return (unsigned)((y0_mag > y1_mag) ? y0_mag : y1_mag);
}

void Texture::standard_filter_orientations(vecD &oris, unsigned n_ori)
{
	oris.resize(n_ori);
	double ori = 0;
	double ori_step = (n_ori > 0) ? (M_PI / (double)n_ori) : 0;
	for (unsigned n = 0; n < n_ori; n++, ori += ori_step)
		oris[n] = ori;
}

Mat Texture::rotate_2D_crop(Mat &m, double ori, unsigned size_x_dst, unsigned size_y_dst)
{
	const int size_x_src = m.cols;
	const int size_y_src = m.rows;
	CV_Assert(m.type() == CV_64F && size_x_dst > 0 && size_y_dst > 0);

	Mat m_rot(size_y_dst, size_x_dst, CV_64F);
	const double cos_ori = cos(ori);
	const double sin_ori = sin(ori);
	const double origin_x_src = (size_x_src - 1) / 2.0;
	const double origin_y_src = (size_y_src - 1) / 2.0;
	double u = -(double)(size_x_dst - 1) / 2.0;
	for (unsigned dst_x = 0; dst_x < size_x_dst; dst_x++) {
		double v = -(double)(size_y_dst - 1) / 2.0;
		for (unsigned dst_y = 0; dst_y < size_y_dst; dst_y++) {
			double x = u*cos_ori + v*sin_ori + origin_x_src;
			double y = v*cos_ori - u*sin_ori + origin_y_src;
			if (x >= 0 && y >= 0) {
				unsigned x0 = (unsigned)floor(x);
				unsigned x1 = (unsigned)ceil(x);
				unsigned y0 = (unsigned)floor(y);
				unsigned y1 = (unsigned)ceil(y);
				if (x0 > 0 && x1 < (unsigned)size_x_src && y0 > 0 && y1 < (unsigned)size_y_src) {
					double dist_x0 = x - x0;
					double dist_x1 = x1 - x;
					double dist_y0 = y - y0;
					double dist_y1 = y1 - y;
					const double m00 = m.at<double>(y0, x0);
					const double m01 = m.at<double>(y1, x0);
					const double m10 = m.at<double>(y0, x1);
					const double m11 = m.at<double>(y1, x1);
					const double t0 = (x0 != x1) ? (dist_x1*m00 + dist_x0*m10) : m00;
					const double t1 = (x0 != x1) ? (dist_x1*m01 + dist_x0*m11) : m01;
					m_rot.at<double>(dst_y, dst_x) = (y0 != y1) ? (dist_y1*t0 + dist_y0*t1) : t0;
				}
			}
			v++;
		}
		u++;
	}

	return m_rot;
}

void Texture::gaussian(vecD &filter, double sigma, unsigned deriv, unsigned support)
{
	const double sigma2_inv = 1.0 / (sigma*sigma);
	const double neg_two_sigma2_inv = -0.5*sigma2_inv;
	unsigned size = 2 * support + 1;
	filter.resize(size);
	double x = -((double)support);
	switch (deriv) {
	case 0:
		for (unsigned n = 0; n < size; n++, x++)
			filter[n] = exp(x*x*neg_two_sigma2_inv);
		break;
	case 1:
		for (unsigned n = 0; n < size; n++, x++)
			filter[n] = exp(x*x*neg_two_sigma2_inv)*(-x);
		break;
	case 2:
		for (unsigned n = 0; n < size; n++, x++)
			filter[n] = exp(x*x*neg_two_sigma2_inv)*(x*x*sigma2_inv - 1);
		break;
	default:
		cout << "Warning: only derivatives 0, 1, 2 supported." << endl;
		break;
	}
	if (deriv > 0) {
		double mean = accumulate(filter.begin(), filter.end(), 0.0) / (double)size;
		for (unsigned n = 0; n < size; n++)
			filter[n] -= mean;
	}
	double sum = 0;
	for (unsigned n = 0; n < size; n++)
		sum += abs(filter[n]);
	for (unsigned n = 0; n < size; n++)
		filter[n] /= sum;
}

void Texture::gaussian_2D(Mat &filter, double sigma_x, double sigma_y, double ori, unsigned deriv, unsigned support_x, unsigned support_y)
{
	unsigned support_x_rot = support_x_rotated(support_x, support_y, -ori);
	unsigned support_y_rot = support_y_rotated(support_x, support_y, -ori);
	vecD fx, fy;
	gaussian(fx, sigma_x, 0,     support_x_rot);
	gaussian(fy, sigma_y, deriv, support_y_rot);

	const int fx_size = fx.size();
	const int fy_size = fy.size();
	filter.create(fy_size, fx_size, CV_64F);
	for (int n_y = 0; n_y < fy_size; n_y++) {
		double *dP = filter.ptr<double>(n_y);
		for (int n_x = 0; n_x < fx_size; n_x++)
			dP[n_x] = fx[n_x] * fy[n_y];
	}
	filter = rotate_2D_crop(filter, ori, 2 * support_x + 1, 2 * support_y + 1);
	if (deriv > 0)
		filter -= mean(filter)[0];
	filter /= sum(abs(filter))[0];
}

void Texture::gaussian_cs_2D(Mat &filter, double sigma_x, double sigma_y, double ori, double scale_factor, unsigned support_x, unsigned support_y)
{
	double sigma_x_c = sigma_x / scale_factor;
	double sigma_y_c = sigma_y / scale_factor;
	Mat m_center, m_surround;
	gaussian_2D(m_center, sigma_x_c, sigma_y_c, ori, 0, support_x, support_y);
	gaussian_2D(m_surround, sigma_x, sigma_y, ori, 0, support_x, support_y);
	filter = m_surround - m_center;
	filter -= mean(filter);
	filter /= sum(abs(filter))[0];
}

void Texture::gaussian_filters(vecM &filters, unsigned n_ori, double sigma, unsigned deriv, double elongation)
{
	vecD oris;
	standard_filter_orientations(oris, n_ori);
	unsigned support = (unsigned)ceil(3 * sigma);
	double sigma_x = sigma;
	double sigma_y = sigma_x / elongation;

	filters.resize(n_ori);
	for (unsigned n = 0; n < n_ori; n++)
		gaussian_2D(filters[n], sigma_x, sigma_y, oris[n], deriv, support, support);
}

void Texture::oe_filters(vecM &filters, unsigned n_ori, double sigma)
{
	gaussian_filters(filters, n_ori, sigma, 2, 3.0);
}

void Texture::texture_filters(vecM &filters, unsigned n_ori, double sigma)
{
	oe_filters(filters, n_ori, sigma);
	
	Mat f_cs;
	unsigned support = (unsigned)ceil(3 * sigma);
	gaussian_cs_2D(f_cs, sigma, sigma, 0, sqrt(2), support, support);

	filters.push_back(f_cs);
}

void Texture::gradient1u(Mat &img1u, Mat &mag1u)
{
	CV_Assert(img1u.type() == CV_8U);
	const int H = img1u.rows, W = img1u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++) {
		Ix.at<int>(y, 0) = abs(img1u.at<byte>(y, 1) - img1u.at<byte>(y, 0)) * 2;
		Ix.at<int>(y, W - 1) = abs(img1u.at<byte>(y, W - 1) - img1u.at<byte>(y, W - 2)) * 2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = abs(img1u.at<byte>(1, x) - img1u.at<byte>(0, x)) * 2;
		Iy.at<int>(H - 1, x) = abs(img1u.at<byte>(H - 1, x) - img1u.at<byte>(H - 2, x)) * 2;
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++) {
		int *xP = Ix.ptr<int>(y);
		const byte *dataP = img1u.ptr<byte>(y);
		for (int x = 1; x < W - 1; x++)
			xP[x] = abs(dataP[x - 1] - dataP[x + 1]);
	}
	for (int y = 1; y < H - 1; y++) {
		int *yP = Iy.ptr<int>(y);
		const byte *tp = img1u.ptr<byte>(y - 1);
		const byte *bp = img1u.ptr<byte>(y + 1);
		for (int x = 0; x < W; x++)
			yP[x] = abs(tp[x] - bp[x]);
	}
	
	mag1u.create(H, W, CV_8U);
	for (int r = 0; r < H; r++) {
		const int *x = Ix.ptr<int>(r), *y = Iy.ptr<int>(r);
		byte *m = mag1u.ptr<byte>(r);
		for (int c = 0; c < W; c++)
			m[c] = min(x[c] + y[c], 255);   //((int)sqrt(sqr(x[c]) + sqr(y[c])), 255);
	}
}

void Texture::texture_gradient(Mat &img3u, vecM &filters, vecM &mags)
{
	Mat gray;
	cvtColor(img3u, gray, COLOR_BGR2GRAY);
	const int size = filters.size();
	mags.resize(size);
	vecM textures(size);
	for (int i = 0; i < size; i++) {
		filter2D(gray, textures[i], CV_8U, filters[i]);
		gradient1u(textures[i], mags[i]);
	}
}

void Texture::texture_gradient(Mat &img3u, vecM &filters, Mat &mag1u)
{
	Mat gray;
	cvtColor(img3u, gray, COLOR_BGR2GRAY);
	const int size = filters.size();
	vecM textures(size), mags(size);
	for (int i = 0; i < size; i++) {
		filter2D(gray, textures[i], CV_8U, filters[i]);
		gradient1u(textures[i], mags[i]);
	}

	mag1u.create(img3u.size(), CV_8U);
	mag1u = Scalar::all(0);
	for (int r = 0; r < img3u.rows; r++)
	for (int c = 0; c < img3u.cols; c++) {
		for (int i = 0; i < size - 1; i++)
		if (mag1u.at<uchar>(r, c) < mags[i].at<uchar>(r, c))
			mag1u.at<uchar>(r, c) = mags[i].at<uchar>(r, c);
	}
}

