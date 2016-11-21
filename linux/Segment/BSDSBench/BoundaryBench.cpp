
#include "../Stdafx.h"
#include "Bench.h"

void seg2bdry(Mat &seg, Mat &bdry, CStr fmt)
{
	CV_Assert_(fmt == "imageSize" || fmt == "doubleSize", "possible values for fmt are: imageSize and doubleSize");
	const int tx = seg.rows;
	const int ty = seg.cols;
	const int nch = seg.channels();
	CV_Assert_(nch == 1, "seg must be a scalar image");

	bdry.create(2 * tx + 1, 2 * ty + 1, CV_8U);
	bdry = Scalar::all(0);

	Mat edgels_v(tx, ty, CV_8U, Scalar(0)), edgels_h(tx, ty, CV_8U, Scalar(0));
	for (int i = 0; i < tx - 1; i++) {
		uchar *ptr_res = edgels_v.ptr<uchar>(i);
		ushort *ptr_curr = seg.ptr<ushort>(i);
		ushort *ptr_next = seg.ptr<ushort>(i + 1);
		for (int j = 0; j < ty; j++)
			ptr_res[j] = (ptr_curr[j] != ptr_next[j]) ? 1 : 0;
	}
	for (int i = 0; i < tx; i++) {
		uchar *ptr_res = edgels_h.ptr<uchar>(i);
		ushort *ptr_curr = seg.ptr<ushort>(i);
		ushort *ptr_next = seg.ptr<ushort>(i);
		for (int j = 0; j < ty - 1; j++)
			ptr_res[j] = (ptr_curr[j] != ptr_next[j + 1]) ? 1 : 0;
	}

	for (int i1 = 2, i2 = 0; i2 < tx; i1 += 2, i2++) {
		uchar *ptr_bdry = bdry.ptr<uchar>(i1);
		uchar *ptr_edgv = edgels_v.ptr<uchar>(i2);
		for (int j1 = 1, j2 = 0; j2 < ty; j1 += 2, j2++)
			ptr_bdry[j1] = ptr_edgv[j2];
	}
	for (int i1 = 1, i2 = 0; i2 < tx; i1 += 2, i2++) {
		uchar *ptr_bdry = bdry.ptr<uchar>(i1);
		uchar *ptr_edgh = edgels_h.ptr<uchar>(i2);
		for (int j1 = 2, j2 = 0; j2 < ty; j1 += 2, j2++)
			ptr_bdry[j1] = ptr_edgh[j2];
	}
	for (int i1 = 2, i2 = 0; i2 < tx - 1; i1 += 2, i2++) {
		uchar *ptr_bdry = bdry.ptr<uchar>(i1);
		uchar *ptr_edgh_curr = edgels_h.ptr<uchar>(i2);
		uchar *ptr_edgh_next = edgels_h.ptr<uchar>(i2 + 1);
		uchar *ptr_edgv = edgels_v.ptr<uchar>(i2);
		for (int j1 = 2, j2 = 0; j2 < ty - 1; j1 += 2, j2++) {
			uchar m1 = max(ptr_edgh_curr[j2], ptr_edgh_next[j2]);
			uchar m2 = max(ptr_edgv[j2], ptr_edgv[j2 + 1]);
			ptr_bdry[j1] = max(m1, m2);
		}
	}

	const int dx = 2 * tx + 1, dy = 2 * ty + 1;
	for (int i = 0; i < dy; i++)
		bdry.at<uchar>(0, i) = bdry.at<uchar>(1, i);
	for (int i = 0; i < dx; i++)
		bdry.at<uchar>(i, 0) = bdry.at<uchar>(i, 1);
	for (int i = 0; i < dy; i++)
		bdry.at<uchar>(dx - 1, i) = bdry.at<uchar>(dx - 2, i);
	for (int i = 0; i < dx; i++)
		bdry.at<uchar>(i, dy - 1) = bdry.at<uchar>(i, dy - 2);

	if (fmt == "imageSize") {
		Mat tmp(tx, ty, CV_8U);
		for (int i1 = 2, i2 = 0; i2 < tx; i1 += 2, i2++) {
			uchar *ptr_bdry = bdry.ptr<uchar>(i1);
			uchar *ptr_tmp = tmp.ptr<uchar>(i2);
			for (int j1 = 2, j2 = 0; j2 < ty; j1 += 2, j2++)
				ptr_tmp[j2] = ptr_bdry[j1];
		}
		tmp.copyTo(bdry);
	}
}

void evaluation_bdry_image(CStr inFile, CStr gtFile, CStr prFile, vecI &thresh, vecF &cntR,
	vecF &sumR, vecF &cntP, vecF &sumP, int nthresh, float maxDist, bool thinpb)
{
	vecM segs, groundTruth;
	matRead(inFile, segs);
	matRead(gtFile, groundTruth);

	nthresh = segs.size();
	thresh.resize(nthresh);
	for (int i = 0; i < nthresh; i++)
		thresh[i] = i + 1;
	cntR.resize(nthresh, 0);  sumR.resize(nthresh, 0);
	cntP.resize(nthresh, 0);  sumP.resize(nthresh, 0);
	
	for (int t = 0; t < nthresh; t++) {
		Mat bmap;
		seg2bdry(segs[t], bmap, "imageSize");
		if (thinpb)
			thinning(bmap);
		Mat accp(bmap.size(), CV_8U, Scalar(0));
		const int height = bmap.rows, width = bmap.cols;

		for (int i = 0; i < groundTruth.size(); i++) {
			Mat match1, match2;
			double cost, oc;
			correspondPixels(match1, match2, cost, oc, bmap, groundTruth[i], maxDist);

			int sum_gt = 0, sum_m2 = 0;
			for (int r = 0; r < height; r++) {
				int *ptr_m1 = match1.ptr<int>(r);
				int *ptr_m2 = match2.ptr<int>(r);
				uchar *ptr_gt = groundTruth[i].ptr<uchar>(r);
				uchar *ptr_ap = accp.ptr<uchar>(r);
				for (int c = 0; c < width; c++) {
					ptr_ap[c] = (ptr_ap[c] || ptr_m1[c]) ? 1 : 0;
					sum_gt += ptr_gt[c];
					sum_m2 += ((ptr_m2[c] > 0) ? 1 : 0);
				}
			}
			sumR[t] += sum_gt;
			cntR[t] += sum_m2;
		}

		int sum_bp = 0, sum_ap = 0;
		for (int r = 0; r < height; r++) {
			uchar *ptr_bp = bmap.ptr<uchar>(r);
			uchar *ptr_ap = accp.ptr<uchar>(r);
			for (int c = 0; c < width; c++) {
				sum_bp += ptr_bp[c];
				sum_ap += ptr_ap[c];
			}
		}
		sumP[t] += sum_bp;
		cntP[t] += sum_ap;
	}

	FILE *file = fopen(_S(prFile), "wb");
	CV_Assert_(file != NULL, "Could not open file for writing...");
	fwrite(&nthresh, sizeof(int), 1, file);
	fwrite(thresh.data(), sizeof(int), nthresh, file);
	fwrite(cntR.data(), sizeof(float), nthresh, file);
	fwrite(sumR.data(), sizeof(float), nthresh, file);
	fwrite(cntP.data(), sizeof(float), nthresh, file);
	fwrite(sumP.data(), sizeof(float), nthresh, file);
	fclose(file);
}

void fmeasure(float *r, float *p, float *f, int size)
{
	for (int i = 0; i < size; i++) {
		if (abs(p[i] + r[i]) <= DOUBLE_EPS)
			f[i] = 2 * p[i] * r[i];
		else
			f[i] = 2 * p[i] * r[i] / (p[i] + r[i]);
	}
}

void maxF(vecI &thresh, vecF &R, vecF &P, float *bestT, float *bestR, float *bestP, float *bestF)
{
	*bestT = (float)thresh[0];
	*bestR = R[0];
	*bestP = P[0];
	fmeasure(&R[0], &P[0], bestF, 1);

	vecF d(100);
	for (int i = 0; i < 100; i++)
		d[i] = 1.0f / 99 * i;

	const int nthresh = thresh.size();
	for (int i = 1; i < nthresh; i++) {
		for (int j = 0; j < 100; j++) {
			float t = thresh[i] * d[j] + thresh[i - 1] * (1 - d[j]);
			float r = R[i] * d[j] + R[i - 1] * (1 - d[j]);
			float p = P[i] * d[j] + P[i - 1] * (1 - d[j]);
			float f;  fmeasure(&r, &p, &f, 1);
			if (f > *bestF)
				*bestT = t, *bestR = r, *bestP = p, *bestF = f;
		}
	}
}

void collect_eval_bdry(CStr pbDir, vecS iids)
{
	//if (GetFileAttributesA(_S(pbDir + "eval_bdry.txt")) != INVALID_FILE_ATTRIBUTES)
		//return;

	int nthresh, s_num = iids.size();
	FILE *file = fopen(_S(pbDir + iids[0] + "_ev1.bin"), "rb");
	int res = fread(&nthresh, sizeof(int), 1, file);
	fclose(file);
	vecI thresh(nthresh);
	vecF cntR(nthresh), sumR(nthresh), cntP(nthresh), sumP(nthresh);
	vecF cntR_total(nthresh, 0), sumR_total(nthresh, 0);
	vecF cntP_total(nthresh, 0), sumP_total(nthresh, 0);
	float cntR_max, sumR_max, cntP_max, sumP_max;
	cntR_max = sumR_max = cntP_max = sumP_max = 0;
	Mat scores(s_num, 5, CV_32F);
	vecF R(nthresh), P(nthresh);
	float bestT, bestR, bestP, bestF;

	for (int i = 0; i < s_num; i++) {
		file = fopen(_S(pbDir + iids[i] + "_ev1.bin"), "rb");
		res = fread(&nthresh, sizeof(int), 1, file);
		res = fread(thresh.data(), sizeof(int), nthresh, file);
		res = fread(cntR.data(), sizeof(float), nthresh, file);
		res = fread(sumR.data(), sizeof(float), nthresh, file);
		res = fread(cntP.data(), sizeof(float), nthresh, file);
		res = fread(sumP.data(), sizeof(float), nthresh, file);
		fclose(file);

		for (int k = 0; k < nthresh; k++) {
			R[k] = (abs(sumR[k]) < DOUBLE_EPS) ? cntR[k] : (cntR[k] / sumR[k]);
			P[k] = (abs(sumP[k]) < DOUBLE_EPS) ? cntP[k] : (cntP[k] / sumP[k]);
		}
		float *F = new float[nthresh];
		fmeasure(R.data(), P.data(), F, nthresh);

		maxF(thresh, R, P, &bestT, &bestR, &bestP, &bestF);
		float *ptr_s = scores.ptr<float>(i);
		ptr_s[0] = (float)(i + 1), ptr_s[1] = bestT, ptr_s[2] = bestR;
		ptr_s[3] = bestP, ptr_s[4] = bestF;

		for (int k = 0; k < nthresh; k++) {
			cntR_total[k] += cntR[k], sumR_total[k] += sumR[k];
			cntP_total[k] += cntP[k], sumP_total[k] += sumP[k];
		}

		int ind = 0; float ff = F[0]; 
		for (int k = 1; k < nthresh; k++) {
			if (F[k] > ff)
				ff = F[k], ind = k;
		}
		cntR_max += cntR[ind], sumR_max += sumR[ind];
		cntP_max += cntP[ind], sumP_max += sumP[ind];
	}

	for (int k = 0; k < nthresh; k++) {
		R[k] = (abs(sumR_total[k]) < DOUBLE_EPS) ? cntR_total[k] : (cntR_total[k] / sumR_total[k]);
		P[k] = (abs(sumP_total[k]) < DOUBLE_EPS) ? cntP_total[k] : (cntP_total[k] / sumP_total[k]);
	}
	float *F = new float[nthresh];
	fmeasure(R.data(), P.data(), F, nthresh);
	maxF(thresh, R, P, &bestT, &bestR, &bestP, &bestF);

	file = fopen(_S(pbDir + "eval_bdry_img.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing...");
	fprintf(file, "%d\n", s_num);
	for (int i = 0; i < s_num; i++) {
		float *ptr_s = scores.ptr<float>(i);
		fprintf(file, "%10d %10f %10f %10f %10f\n", (int)round(ptr_s[0]), ptr_s[1],
			ptr_s[2], ptr_s[3], ptr_s[4]);
	}
	fclose(file);

	file = fopen(_S(pbDir + "eval_bdry_thr.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing...");
	fprintf(file, "%d\n", nthresh);
	for (int i = 0; i < nthresh; i++)
		fprintf(file, "%10f %10f %10f %10f\n", (float)thresh[i], R[i], P[i], F[i]);
	fclose(file);

	float R_max = (abs(sumR_max) < DOUBLE_EPS) ? cntR_max : (cntR_max / sumR_max);
	float P_max = (abs(sumP_max) < DOUBLE_EPS) ? cntP_max : (cntP_max / sumP_max);
	float F_max;  fmeasure(&R_max, &P_max, &F_max, 1);

	vecI indR; vecF Ru, Pu, Ri(101);
	unique(R, Ru, indR);
	Pu.resize(indR.size());
	for (int i = 0; i < indR.size(); i++)
		Pu[i] = P[indR[i]];
	for (int i = 0; i < 101; i++)
		Ri[i] = 0.01f*i;
	float Area_PR = 0;
	if (Ru.size() > 1) {
		vecF P_int1; interp1(Ru, Pu, Ri, P_int1);
		Area_PR = accumulate(P_int1.begin(), P_int1.end(), 0.0f)*0.01f;
	}

	file = fopen(_S(pbDir + "eval_bdry.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing...");
	fprintf(file, "%10f %10f %10f %10f %10f %10f %10f %10f\n", bestT, bestR, bestP, bestF,
		R_max, P_max, F_max, Area_PR);
	fclose(file);
}

float evaluation_bdry_single_image(Mat &seg, vecM &groundTruth, float maxDist, bool thinpb)
{
	float cntR, sumR, cntP, sumP, R, P, F;
	cntR = sumR = cntP = sumP = 0;

	Mat bmap;
	seg2bdry(seg, bmap, "imageSize");
	if (thinpb)
		thinning(bmap);
	Mat accp(bmap.size(), CV_8U, Scalar(0));
	const int height = bmap.rows, width = bmap.cols;

	for (int i = 0; i < groundTruth.size(); i++) {
		Mat match1, match2;
		double cost, oc;
		correspondPixels(match1, match2, cost, oc, bmap, groundTruth[i], maxDist);

		int sum_gt = 0, sum_m2 = 0;
		for (int r = 0; r < height; r++) {
			int *ptr_m1 = match1.ptr<int>(r);
			int *ptr_m2 = match2.ptr<int>(r);
			uchar *ptr_gt = groundTruth[i].ptr<uchar>(r);
			uchar *ptr_ap = accp.ptr<uchar>(r);
			for (int c = 0; c < width; c++) {
				ptr_ap[c] = (ptr_ap[c] || ptr_m1[c]) ? 1 : 0;
				sum_gt += ptr_gt[c];
				sum_m2 += ((ptr_m2[c] > 0) ? 1 : 0);
			}
		}
		sumR += sum_gt;
		cntR += sum_m2;
	}

	for (int r = 0; r < height; r++) {
		uchar *ptr_bp = bmap.ptr<uchar>(r);
		uchar *ptr_ap = accp.ptr<uchar>(r);
		for (int c = 0; c < width; c++) {
			sumP += ptr_bp[c];
			cntP += ptr_ap[c];
		}
	}

	R = (abs(sumR) < DOUBLE_EPS) ? cntR : (cntR / sumR);
	P = (abs(sumP) < DOUBLE_EPS) ? cntP : (cntP / sumP);
	fmeasure(&R, &P, &F, 1);

	return F;
}
