
#pragma once

#include "../StdAfx.h"

void plot_eval(CStr evalDir, CStr col = "r");
void allBench(vecS iids, CStr gtDir, CStr inDir, CStr outDir, int nthresh = 99, float maxDist = 0.0075, bool thinpb = true);

void seg2bdry(Mat &seg, Mat &bdry, CStr fmt = "doubleSize");
float evaluation_bdry_single_image(Mat &seg, vecM &groundTruth, float maxDist = 0.0075, bool thinpb = true);

void collect_eval_reg(CStr ucmDir);
void collect_eval_bdry(CStr pbDir);
void evaluation_reg_image(CStr inFile, CStr gtFile, CStr evFile2, CStr evFile3, CStr evFile4,
	vecI &thresh, vecF &cntR, vecF &sumR, vecF &cntP, vecF &sumP, float *cntR_best, int nthresh = 99);
void evaluation_bdry_image(CStr inFile, CStr gtFile, CStr prFile, vecI &thresh, vecF &cntR,
	vecF &sumR, vecF &cntP, vecF &sumP, int nthresh = 99, float maxDist = 0.0075, bool thinpb = true);

void correspondPixels(Mat &match1, Mat &match2, double &cost, double &oc, Mat &bmap, Mat &gt, double maxDist = 0.0075, double outlierCost = 100);

bool matWrite(CStr &filename, vector<CMat> &_M);
bool matRead(CStr &filename, vector<Mat> &_M);
void thinning(Mat &im);
void interp1(vecF &x, vecF &y, vecF &x_new, vecF &y_new);
void unique(vecF &v, vecF &vu, vecI &ind);