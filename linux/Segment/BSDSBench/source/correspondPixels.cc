#include "../../Stdafx.h"

#include <string.h>
//#include <mex.h>
#include "Matrix.hh"
#include "csa.hh"
#include "match.hh"

//extern "C" {
//
//static const double maxDistDefault = 0.0075;
//static const double outlierCostDefault = 100;
//
//void mexFunction (
//    int nlhs, mxArray* plhs[],
//    int nrhs, const mxArray* prhs[])
//{
//    // check number of arguments
//    if (nlhs < 2) {
//        mexErrMsgTxt("Too few output arguments.");
//    }
//    if (nlhs > 4) {
//        mexErrMsgTxt("Too many output arguments.");
//    }
//    if (nrhs < 2) {
//        mexErrMsgTxt("Too few input arguments.");
//    }
//    if (nrhs > 4) {
//        mexErrMsgTxt("Too many input arguments.");
//    }
//
//    // get arguments
//    double* bmap1 = mxGetPr(prhs[0]);
//    double* bmap2 = mxGetPr(prhs[1]);
//    const double maxDist = 
//        (nrhs>2) ? mxGetScalar(prhs[2]) : maxDistDefault;
//    const double outlierCost = 
//        (nrhs>3) ? mxGetScalar(prhs[3]) : outlierCostDefault;
//
//    // check arguments
//    if (mxGetM(prhs[0]) != mxGetM(prhs[1]) 
//        || mxGetN(prhs[0]) != mxGetN(prhs[1])) {
//        mexErrMsgTxt("bmap1 and bmap2 must be the same size");
//    }
//    if (maxDist < 0) {
//        mexErrMsgTxt("maxDist must be >= 0");
//    }
//    if (outlierCost <= 1) {
//        mexErrMsgTxt("outlierCost must be >1");
//    }
//
//    // do the computation
//    const int rows = mxGetM(prhs[0]);
//    const int cols = mxGetN(prhs[0]);
//    const double idiag = sqrt( (double)(rows*rows + cols*cols ));
//    const double oc = outlierCost*maxDist*idiag;
//    Matrix m1, m2;
//    const double cost = matchEdgeMaps(
//        Matrix(rows,cols,bmap1), Matrix(rows,cols,bmap2),
//        maxDist*idiag, oc,
//        m1, m2);
//    
//    // set output arguments
//    plhs[0] = mxCreateDoubleMatrix(rows, cols, mxREAL);
//    plhs[1] = mxCreateDoubleMatrix(rows, cols, mxREAL);
//    double* match1 = mxGetPr(plhs[0]);
//    double* match2 = mxGetPr(plhs[1]);
//    memcpy(match1,m1.data(),m1.numel()*sizeof(double));
//    memcpy(match2,m2.data(),m2.numel()*sizeof(double));
//    if (nlhs > 2) { plhs[2] = mxCreateDoubleScalar(cost); }
//    if (nlhs > 3) { plhs[3] = mxCreateDoubleScalar(oc); }
//}
//
//}; // extern "C"

void correspondPixels(Mat &match1, Mat &match2, double &cost, double &oc, Mat &bmap, Mat &gt, double maxDist, double outlierCost)
{
	CV_Assert_(bmap.rows == gt.rows&&bmap.cols == gt.cols, "bmap and gt must be the same size");
	CV_Assert_(maxDist >= 0, "maxDist must be >= 0");
	CV_Assert_(outlierCost > 1, "outlierCost must be >1");

	const int rows = bmap.rows;
	const int cols = bmap.cols;
	const double idiag = sqrt((double)(rows*rows + cols*cols));
	oc = outlierCost*maxDist*idiag;
	Matrix m1, m2;

	Matrix bmap1(rows, cols), bmap2(rows, cols);
	for (int i = 0; i < rows; i++) {
		uchar *bP = bmap.ptr<uchar>(i);
		uchar *gP = gt.ptr<uchar>(i);
		for (int j = 0; j < cols; j++) {
			bmap1(i, j) = (double)bP[j];
			bmap2(i, j) = (double)gP[j];
		}
	}

	cost = matchEdgeMaps(bmap1, bmap2, maxDist*idiag, oc, m1, m2);

	match1.create(rows, cols, CV_32S);
	match2.create(rows, cols, CV_32S);
	for (int i = 0; i < rows; i++) {
		int *m1P = match1.ptr<int>(i);
		int *m2P = match2.ptr<int>(i);
		for (int j = 0; j < cols; j++) {
			m1P[j] = (int)round(m1(i, j));
			m2P[j] = (int)round(m2(i, j));
		}
	}
}
