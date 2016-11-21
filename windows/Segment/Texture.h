
#pragma once

#include "Stdafx.h"

class Texture
{
	static unsigned support_x_rotated(unsigned support_x, unsigned support_y, double ori);
	static unsigned support_y_rotated(unsigned support_x, unsigned support_y, double ori);

	static void standard_filter_orientations(vecD &oris, unsigned n_ori);

	static Mat rotate_2D_crop(Mat &m, double ori, unsigned size_x_dst, unsigned size_y_dst);

public:

	static void gaussian(vecD &filter, double sigma, unsigned deriv, unsigned support);
	static void gaussian_2D(Mat &filter, double sigma_x, double sigma_y, double ori, unsigned deriv, unsigned support_x, unsigned support_y);
	static void gaussian_cs_2D(Mat &filter, double sigma_x, double sigma_y, double ori, double scale_factor, unsigned support_x, unsigned support_y);
	static void gaussian_filters(vecM &filters, unsigned n_ori, double sigma, unsigned deriv, double elongation);
	static void oe_filters(vecM &filters, unsigned n_ori, double sigma);
	static void texture_filters(vecM &filters, unsigned n_ori, double sigma);

	static void gradient1u(Mat &img1u, Mat &mag1u);
	static void texture_gradient(Mat &img3u, vecM &filters, vecM &mags);
	static void texture_gradient(Mat &img3u, vecM &filters, Mat &mag1u);
    
};