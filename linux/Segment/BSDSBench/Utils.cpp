
#include "../Stdafx.h"
#include "Bench.h"

// Write matrix to binary file
bool matWrite(CStr &filename, vecM &_M)
{
	const int matCnt = _M.size();
	vecM M(matCnt);
	for (int i = 0; i < matCnt; i++) {
		M[i].create(_M[i].size(), _M[i].type());
		memcpy(M[i].data, _M[i].data, M[i].step*M[i].rows);
	}
	FILE* file = fopen(_S(filename), "wb");
	if (file == NULL || M[0].empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, file);
	fwrite(&matCnt, sizeof(int), 1, file);
	for (int i = 0; i < matCnt; i++) {
		int headData[3] = { M[i].cols, M[i].rows, M[i].type() };
		fwrite(headData, sizeof(int), 3, file);
		fwrite(M[i].data, sizeof(char), M[i].step * M[i].rows, file);
	}
	std::fclose(file);
	return true;
}

// Read matrix from binary file
bool matRead(CStr &filename, vecM &_M)
{
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int res = fread(buf, sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		std::cout << "Invalidate CvMat data file " << _S(filename) << endl;
		return false;
	}
	int matCnt;
	res = fread(&matCnt, sizeof(int), 1, f);
	_M.resize(matCnt);
	for (int i = 0; i < matCnt; i++) {
		int headData[3]; // MatCnt, Width, Height, Type
		res = fread(headData, sizeof(int), 3, f);
		Mat M(headData[1], headData[0], headData[2]);
		res = fread(M.data, sizeof(char), M.step * M.rows, f);
		M.copyTo(_M[i]);
	}
	std::fclose(f);
	return true;
}

void thinningIteration(Mat &im, int iter)
{
	for (int i = 1; i < im.rows - 1; i++) {
		for (int j = 1; j < im.cols - 1; j++) {
			if (im.at<uchar>(i, j) == 0)
				continue;

			uchar xh = 0, n, n1 = 0, n2 = 0, c3 = 0, x[8];
			x[0] = im.at<uchar>(i, j + 1);
			x[1] = im.at<uchar>(i - 1, j + 1);
			x[2] = im.at<uchar>(i - 1, j);
			x[3] = im.at<uchar>(i - 1, j - 1);
			x[4] = im.at<uchar>(i, j - 1);
			x[5] = im.at<uchar>(i + 1, j - 1);
			x[6] = im.at<uchar>(i + 1, j);
			x[7] = im.at<uchar>(i + 1, j + 1);

			for (int k = 0; k < 4; k++)
				xh += (x[2 * k] == 0 && (x[2 * k + 1] == 1 || x[(2 * k + 2) % 8] == 1)) ? 1 : 0;
			for (int k = 0; k < 4; k++)
				n1 += (x[2 * k] == 1 || x[2 * k + 1] == 1) ? 1 : 0;
			for (int k = 0; k < 4; k++)
				n2 += (x[2 * k + 1] == 1 || x[(2 * k + 2) % 8] == 1) ? 1 : 0;
			n = min(n1, n2);
			if (iter == 0)
				c3 = ((x[0] || x[2] || x[7] == 0) && x[0]) ? 1 : 0;
			else
				c3 = ((x[5] || x[6] || x[3] == 0) && x[4]) ? 1 : 0;

			if (xh == 1 && n >= 2 && n <= 3 && c3 == 1)
				im.at<uchar>(i, j) = 0;
		}
	}
}

void thinning(Mat &im)
{
	CV_Assert_(im.type() == CV_8U, "The type of image must be Int...");
	Mat prev, diff;
	im.copyTo(prev);

	do {
		thinningIteration(im, 0);
		thinningIteration(im, 1);
		absdiff(im, prev, diff);
		im.copyTo(prev);
	} while (countNonZero(diff) > 0);
}

void interp1(vecF &x, vecF &y, vecF &x_new, vecF &y_new)
{
	const int s = x.size();
	vecF slope(s - 1), intercept(s - 1);
	for (int i = 0; i < s - 1; ++i){
		float dx = x[i + 1] - x[i];
		float dy = y[i + 1] - y[i];
		slope[i] = dy / dx;
		intercept[i] = y[i] - x[i] * slope[i];
	}

	float xmin = *min_element(x.begin(), x.end());
	float xmax = *max_element(x.begin(), x.end());
	for (int i = 0; i < x_new.size(); i++) {
		if (x_new[i]<xmin || x_new[i]>xmax) {
			x_new.erase(x_new.begin() + i);
			i--;
		}
	}

	y_new.resize(x_new.size());
	for (int i = 0; i < x_new.size(); ++i) {
		int idx = -1;  double dist = numeric_limits<float>::max();
		for (int j = 0; j < s - 1; j++) {
			double newDist = x_new[i] - x[j];
			if (newDist >= 0 && newDist < dist)
				dist = newDist, idx = j;
		}
		CV_Assert_(idx >= 0 && idx < s - 1, "Error in interp1...");
		y_new[i] = slope[idx] * x_new[i] + intercept[idx];
	}
}

void unique(vecF &v, vecF &vu, vecI &ind)
{
	vector<pair<float, int> > tv(v.size());
	for (int i = 0; i < v.size(); i++)
		tv[i] = make_pair(v[i], i);
	std::sort(tv.begin(), tv.end());
	for (int i = 1; i < tv.size(); i++) {
		if (abs(tv[i].first - tv[i - 1].first) < 1e-6) {
			tv.erase(tv.begin() + i);
			i--;
		}
	}
	vu.resize(tv.size());
	ind.resize(tv.size());
	for (int i = 0; i < tv.size(); i++)
		vu[i] = tv[i].first, ind[i] = tv[i].second;
}
