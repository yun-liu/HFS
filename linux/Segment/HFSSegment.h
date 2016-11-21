
#pragma once

#include "Stdafx.h"
#include "Magnitude.h"
#include "DataSet.h"
#include "FelzenSegment/segment-image.h"

class HFSSegment
{
	DataSet _bsds;
	vecF w1, w2;
	vecM wMat;

	Magnitude mag_engine;

	gSLIC::UChar4Image *in_img, *out_img;
	gSLIC::engines::core_engine *gSLIC_engine;
	void constructEngine();

	// Calculate the image gradient: center option as in VLFeat
	static void gradientLab(CMat &bgr3u, Mat &mag3u);
	static inline Vec3i vecIdtDist(const Vec3b &u, const Vec3b &v) { return Vec3i(abs(u[0] - v[0]), abs(u[1] - v[1]), abs(u[2] - v[2])); }

	// Each row of X is a feature vector, with corresponding label in Y. Return a CV_32F weight Mat
	static Mat trainSVM(CMat &X1f, const vecF &Y, int sT, double C, double bias = -1, double eps = 0.01);
	// X1f is the set of training features, and Y is the set of training labels
	static Mat trainSVM(const vecM &X1f, const vecF &Y, int sT, double C, double bias = -1, double eps = 0.01, int maxTrainNum = 100000);
	
public:
	HFSSegment();
	HFSSegment(DataSet &data);
	~HFSSegment();

	void load_image(const Mat &inimg, gSLIC::UChar4Image *outimg);
	Mat getSLICIdx(Mat &img3u, int &num_css);
	
	inline float getEulerDistance(Vec3f in1, Vec3f in2) {
		return sqrt(square(in1[0] - in2[0]) + square(in1[1] - in2[1]) + square(in1[2] - in2[2]));
	}

	Vec4f getColorFeature(Vec3f &in1, Vec3f &in2);
	int getAvgGradientBdry(Mat &idx_mat, vecM &mag1u, Mat &bd_num, vecM &gradients, int num_css);
	void getVariance3C(Mat &img3u, Mat &idx_mat, Mat &var, int num_css);
	void getHistColor(Mat &img3u, Mat &idx_mat, Mat &hist, int num_css);
	void getHistGradient(Mat &img3u, Mat &idx_mat, Mat &hist, int num_css);
	float getChiHist(Mat &hist1, Mat &hist2);

	void generateTrainData();
	void mergePartial();
	void trainStageI(int trainSample = 2500);
	void trainStageII();
	void trainSegment();

	void loadTrainedModel(CStr path);
	void getSegmentationI(Mat &seg, Mat &lab3u, Mat &mag1u, Mat &idx_mat, int &num_css, float c, int min_size);
	void getSegmentationII(Mat &seg, Mat &img3u, Mat &lab3u, Mat &mag1u, Mat &idx_mat, int &num_css, float c, int min_size);
	void getSegmentationIII(Mat &seg, Mat &hed, Mat &idx_mat, int &num_css, float threshold);
	void drawSegmentationRes(Mat &show, Mat &seg, Mat &img3u, int num_css);

	void runDataSet(float c, int min_size);
	int processImage(Mat &seg, Mat &img3u, float c, int min_size);

	bool matWrite(CStr &filename, vecM &_M);
	bool matRead(CStr &filename, vecM &_M);
};