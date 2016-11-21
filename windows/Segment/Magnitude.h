
#pragma once

#include "Stdafx.h"

class Magnitude
{
	gSLIC::IntImage *delta_x, *delta_y, *mag;
	gSLIC::UCharImage *gray_img, *nms_mag;
	gSLIC::Vector2i img_size;

public:
	Magnitude();
	~Magnitude();

	void load_image(const Mat& inimg, gSLIC::UCharImage* outimg);
	void load_image(const gSLIC::UCharImage* inimg, Mat& outimg);

	void derrivative_x_y();
	void non_max_supp();

	void process_img(Mat &bgr3u, Mat &mag1u);

};