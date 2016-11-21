#include "../Stdafx.h"
#include "Bench.h"

float rand_index(Mat n)
{
	const int _w = n.cols;
	const int _h = n.rows;

	double N = (double)sum(n)[0];
	vecD n_u(_h, 0), n_v(_w, 0);
	for (int i = 0; i < _h; i++)
		n_u[i] = sum(n.row(i))[0];
	for (int i = 0; i < _w; i++)
		n_v[i] = sum(n.col(i))[0];
	double N_choose_2 = N*(N - 1)*0.5;
	
	double sum_u = 0, sum_v = 0, sum_n = 0;
	for (int i = 0; i < _h; i++)
		sum_u += n_u[i] * n_u[i];
	for (int i = 0; i < _w; i++)
		sum_v += n_v[i] * n_v[i];
	for (int i = 0; i < _h; i++) {
		int *data = n.ptr<int>(i);
		for (int j = 0; j < _w; j++)
			sum_n += double(data[j]) * double(data[j]);
	}
	float ri = float(1 - (sum_u / 2 + sum_v / 2 - sum_n) / N_choose_2);
	return ri;
}

Mat log2_quotient(Mat A, Mat B)
{
	CV_Assert(A.size() == B.size() && A.type() == B.type()&&A.type() == CV_32F);
	const int _h = A.rows, _w = A.cols;

	Mat lq(A.size(), CV_32F);
	for (int i = 0; i < _h; i++) {
		float *ptr_A = A.ptr<float>(i);
		float *ptr_B = B.ptr<float>(i);
		float *ptr_lq = lq.ptr<float>(i);
		for (int j = 0; j < _w; j++) {
			float nume = ptr_A[j], deno = ptr_B[j];
			if (abs(ptr_A[j]) <= DOUBLE_EPS)
				nume += ptr_B[j];
			if (abs(ptr_B[j]) <= DOUBLE_EPS)
				nume += 1, deno += 1;
			ptr_lq[j] = log2(nume / deno);
		}
	}
	return lq;
}

float variation_of_information(Mat n)
{
	const int _h = n.rows;
	const int _w = n.cols;

	int N = (int)sum(n)[0];
	Mat joint, mul12(_h, _w, CV_32F);
	n.copyTo(joint);
	joint.convertTo(joint, CV_32F);
	joint /= N;
	vecF marginal_1(_h, 0), marginal_2(_w, 0);
	for (int i = 0; i < _h; i++)
		marginal_1[i] = (float)sum(joint.row(i))[0];
	for (int i = 0; i < _w; i++)
		marginal_2[i] = (float)sum(joint.col(i))[0];

	float H1 = 0, H2 = 0, MI = 0, vi;
	for (int i = 0; i < _h; i++)
		H1 -= marginal_1[i] * ((abs(marginal_1[i]) <= DOUBLE_EPS) ? log2(marginal_1[i] + 1) : log2(marginal_1[i]));
	for (int i = 0; i < _w; i++)
		H2 -= marginal_2[i] * ((abs(marginal_2[i]) <= DOUBLE_EPS) ? log2(marginal_2[i] + 1) : log2(marginal_2[i]));

	for (int i = 0; i < _h; i++) {
		float *ptr = mul12.ptr<float>(i);
		for (int j = 0; j < _w; j++)
			ptr[j] = marginal_1[i] * marginal_2[j];
	}
	Mat lq = log2_quotient(joint, mul12);
	MI = (float)sum(joint.mul(lq))[0];
	vi = H1 + H2 - 2 * MI;
	return vi;
}

void match_segmentations2(Mat &seg, vecM &groundTruth, float *sumRI, float *sumVOI)
{
	const int _h = seg.rows;
	const int _w = seg.cols;
	const int _n = groundTruth.size();
	*sumRI = *sumVOI = 0;

	for (int s = 0; s < _n; s++) {
		Mat &gt = groundTruth[s];
		int num1 = -1, num2 = -1;
		for (int i = 0; i < _h; i++) {
			ushort *ptr_seg = seg.ptr<ushort>(i);
			ushort *ptr_gt = gt.ptr<ushort>(i);
			for (int j = 0; j < _w; j++) {
				if (ptr_seg[j] > num1)
					num1 = ptr_seg[j];
				if (ptr_gt[j] > num2)
					num2 = ptr_gt[j];
			}
		}

		Mat confcounts = Mat::zeros(num1, num2, CV_32S);
		for (int i = 0; i < _h; i++) {
			ushort *ptr_seg = seg.ptr<ushort>(i);
			ushort *ptr_gt = gt.ptr<ushort>(i);
			for (int j = 0; j < _w; j++)
				confcounts.at<int>(ptr_seg[j] - 1, ptr_gt[j] - 1)++;
		}

		*sumRI += rand_index(confcounts);
		*sumVOI += variation_of_information(confcounts);
	}

	*sumRI /= _n;
	*sumVOI /= _n;
}

void match_segmentations(Mat &seg, vecM &groundTruth, Mat &matches)
{
	const int _h = seg.rows;
	const int _w = seg.cols;
	const int _n = groundTruth.size();

	int total_gt = 0, max_seg = -1;
	vecI max_gts(_n, -1);
	for (int s = 0; s < _n; s++) {
		Mat &gt = groundTruth[s];
		for (int i = 0; i < _h; i++) {
			ushort *ptr_gt = gt.ptr<ushort>(i);
			for (int j = 0; j < _w; j++) {
				if (ptr_gt[j] > max_gts[s])
					max_gts[s] = ptr_gt[j];
			}
		}
		total_gt += max_gts[s];
	}
	for (int i = 0; i < _h; i++) {
		ushort *ptr_seg = seg.ptr<ushort>(i);
		for (int j = 0; j < _w; j++) {
			if (ptr_seg[j] > max_seg)
				max_seg = ptr_seg[j];
		}
	}

	int cnt = 0;
	matches = Mat::zeros(total_gt, max_seg, CV_32F);
	for (int s = 0; s < _n; s++) {
		Mat &gt = groundTruth[s];
		int num1 = max_gts[s] + 1, num2 = max_seg + 1;
		Mat confcounts = Mat::zeros(num1, num2, CV_32S);

		vecI hs(num1*num2, 0);
		for (int i = 0; i < _h; i++) {
			ushort *ptr_gt = gt.ptr<ushort>(i);
			ushort *ptr_seg = seg.ptr<ushort>(i);
			for (int j = 0; j < _w; j++) {
				int t = 1 + ptr_gt[j] + num1*ptr_seg[j];
				hs[t - 1]++;
			}
		}
		for (int i = 0; i < num1; i++) {
			int *ptr_conf = confcounts.ptr<int>(i);
			for (int j = 0; j < num2; j++)
				ptr_conf[j] += hs[j*num1 + i];
		}
		
		Mat accuracies(num1, num2, CV_32F);
		vecI sum_row(num2, 0), sum_col(num1, 0);
		for (int i = 0; i < num1; i++) {
			int *ptr_conf = confcounts.ptr<int>(i);
			for (int j = 0; j < num2; j++) {
				sum_row[j] += ptr_conf[j];
				sum_col[i] += ptr_conf[j];
			}
		}
		for (int i = 0; i < num1; i++) {
			int *ptr_conf = confcounts.ptr<int>(i);
			float *ptr_accu = accuracies.ptr<float>(i);
			for (int j = 0; j < num2; j++) {
				ptr_accu[j] = (float)ptr_conf[j] / (sum_col[i] + sum_row[j] - ptr_conf[j]);
			}
		}

		for (int i = cnt; i < cnt + num1 - 1; i++) {
			float *ptr_mat = matches.ptr<float>(i);
			float *ptr_accu = accuracies.ptr<float>(i - cnt + 1);
			for (int j = 0; j < num2 - 1; j++)
				ptr_mat[j] = ptr_accu[j + 1];
		}
		cnt += max_gts[s];
	}
	matches = matches.t();
}

void evaluation_reg_image(CStr inFile, CStr gtFile, CStr evFile2, CStr evFile3, CStr evFile4,
	vecI &thresh, vecF &cntR, vecF &sumR, vecF &cntP, vecF &sumP, float *cntR_best, int nthresh)
{
	vecM segs, groundTruth;
	matRead(inFile, segs);
	matRead(gtFile, groundTruth);
	const int nsegs = groundTruth.size();
	CV_Assert_(nsegs != 0, "bad gtFile!");
	nthresh = segs.size();
	thresh.resize(nthresh);
	for (int i = 0; i < nthresh; i++)
		thresh[i] = i + 1;

	const int _h = groundTruth[0].rows;
	const int _w = groundTruth[0].cols;

	int total_gt = 0;
	vecI max_gts(nsegs, -1);
	for (int s = 0; s < nsegs; s++) {
		for (int i = 0; i < _h; i++) {
			ushort *ptr_gt = groundTruth[s].ptr<ushort>(i);
			for (int j = 0; j < _w; j++)
			    if (ptr_gt[j] > max_gts[s])
				   max_gts[s] = ptr_gt[j];
		}
		total_gt += max_gts[s];
	}
	vecI regionsGT(total_gt, 0);
	int cnt = 0;
	for (int s = 0; s < nsegs; s++) {
		for (int i = 0; i < _h; i++) {
			ushort *ptr_gt = groundTruth[s].ptr<ushort>(i);
			for (int j = 0; j < _w; j++)
				regionsGT[cnt + ptr_gt[j] - 1]++;
		}
		cnt += max_gts[s];
	}
	
	cntR.resize(nthresh, 0);
	sumR.resize(nthresh, 0);
	cntP.resize(nthresh, 0);
	sumP.resize(nthresh, 0);
	vecF sumRI(nthresh, 0), sumVOI(nthresh, 0), best_matchesGT(total_gt, 0);

	for (int t = 0; t < nthresh; t++) {
		match_segmentations2(segs[t], groundTruth, &sumRI[t], &sumVOI[t]);
		Mat matches;
		match_segmentations(segs[t], groundTruth, matches);
		vecF matchesSeg(matches.rows, -1), matchesGT(matches.cols, -1);
		for (int i = 0; i < matches.rows; i++) {
			float *ptr_mat = matches.ptr<float>(i);
			for (int j = 0; j < matches.cols; j++) {
				if (matchesSeg[i] < ptr_mat[j])
					matchesSeg[i] = ptr_mat[j];
				if (matchesGT[j] < ptr_mat[j])
					matchesGT[j] = ptr_mat[j];
			}
		}

		int max_seg = -1;
		for (int i = 0; i < _h; i++) {
			ushort *ptr_seg = segs[t].ptr<ushort>(i);
			for (int j = 0; j < _w; j++)
				if (ptr_seg[j] > max_seg)
					max_seg = ptr_seg[j];
		}
		vecI regionsSeg(max_seg, 0);
		for (int i = 0; i < _h; i++) {
			ushort *ptr_seg = segs[t].ptr<ushort>(i);
			for (int j = 0; j < _w; j++)
				regionsSeg[ptr_seg[j] - 1]++;
		}

		for (int r = 0; r < max_seg; r++) {
			cntP[t] += regionsSeg[r] * matchesSeg[r];
			sumP[t] += regionsSeg[r];
		}
		for (int r = 0; r < total_gt; r++) {
			cntR[t] += regionsGT[r] * matchesGT[r];
			sumR[t] += regionsGT[r];
		}

		for (int i = 0; i < total_gt; i++) {
			if (best_matchesGT[i] < matchesGT[i])
				best_matchesGT[i] = matchesGT[i];
		}
	}
	
	*cntR_best = 0;
	for (int r = 0; r < total_gt; r++)
		*cntR_best = *cntR_best + regionsGT[r] * best_matchesGT[r];
	
	FILE* file = fopen(_S(evFile2), "wb");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fwrite(&nthresh, sizeof(int), 1, file);
	fwrite(thresh.data(), sizeof(int), thresh.size(), file);
	fwrite(cntR.data(), sizeof(float), cntR.size(), file);
	fwrite(sumR.data(), sizeof(float), sumR.size(), file);
	fwrite(cntP.data(), sizeof(float), cntP.size(), file);
	fwrite(sumP.data(), sizeof(float), sumP.size(), file);
	std::fclose(file);

	file = NULL;
	file = fopen(_S(evFile3), "wb");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fwrite(cntR_best, sizeof(float), 1, file);
	std::fclose(file);

	file = NULL;
	file = fopen(_S(evFile4), "wb");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fwrite(&nthresh, sizeof(int), 1, file);
	fwrite(thresh.data(), sizeof(int), thresh.size(), file);
	fwrite(sumRI.data(), sizeof(float), sumRI.size(), file);
	fwrite(sumVOI.data(), sizeof(float), sumVOI.size(), file);
	std::fclose(file);
}

void collect_eval_reg(CStr ucmDir, vecS iids)
{
	//if (GetFileAttributesA(_S(ucmDir + "eval_cover.txt")) != INVALID_FILE_ATTRIBUTES)
		//return;

	FILE *file = fopen(_S(ucmDir + iids[0] + "_ev2.bin"), "rb");
	int nthresh, s_num = iids.size();
	int res = fread(&nthresh, sizeof(int), 1, file);
	fclose(file);
	vecF cntR_total(nthresh, 0), sumR_total(nthresh, 0);
	vecF cntP_total(nthresh, 0), sumP_total(nthresh, 0);
	float cntR_best = 0, sumR_best = 0, cntP_best = 0, sumP_best = 0;
	vecF globalRI(nthresh, 0), globalVOI(nthresh, 0);
	float RI_best = 0, VOI_best = 0;
	float cntR_best_total = 0;
	Mat scores1(s_num, 2, CV_32S, Scalar(0));
	Mat scores2(s_num, 2, CV_32F, Scalar(0));
	vecI thresh(nthresh);

	for (int i = 0; i < s_num; i++) {
		vecF cntR(nthresh), sumR(nthresh);
		vecF cntP(nthresh), sumP(nthresh);
		file = fopen(_S(ucmDir + iids[i] + "_ev2.bin"), "rb");
		res = fread(&nthresh, sizeof(int), 1, file);
		res = fread(thresh.data(), sizeof(int), nthresh, file);
		res = fread(cntR.data(), sizeof(float), nthresh, file);
		res = fread(sumR.data(), sizeof(float), nthresh, file);
		res = fread(cntP.data(), sizeof(float), nthresh, file);
		res = fread(sumP.data(), sizeof(float), nthresh, file);
		std::fclose(file);

		vecF R(nthresh), P(nthresh);
		for (int t = 0; t < nthresh; t++) {
			if (abs(sumR[t]) <= DOUBLE_EPS)
				R[t] = cntR[t];
			else
				R[t] = cntR[t] / sumR[t];
			if (abs(sumP[t]) <= DOUBLE_EPS)
				P[t] = cntP[t];
			else
				P[t] = cntP[t] / sumP[t];
		}

		float bestR = -10000;
		int ind;
		for (int t = 0; t < nthresh; t++) {
			if (R[t] > bestR) {
				bestR = R[t];
				ind = t;
			}
		}
		int bestT = thresh[ind];
		float bestP = P[ind];
		int *ptr_scores1 = scores1.ptr<int>(i);
		float *ptr_scores2 = scores2.ptr<float>(i);
		ptr_scores1[0] = i + 1, ptr_scores1[1] = bestT;
		ptr_scores2[0] = bestR, ptr_scores2[1] = bestP;

		for (int t = 0; t < nthresh; t++) {
			cntR_total[t] += cntR[t];
			sumR_total[t] += sumR[t];
			cntP_total[t] += cntP[t];
			sumP_total[t] += sumP[t];
		}

		bestR = -10000, ind = -1;
		for (int t = 0; t < nthresh; t++) {
			if (R[t] >= bestR) {
				bestR = R[t];
				ind = t;
			}
		}
		cntR_best += cntR[ind], sumR_best += sumR[ind];
		cntP_best += cntP[ind], sumP_best += sumP[ind];

		file = fopen(_S(ucmDir + iids[i] + "_ev3.bin"), "rb");
		float tmp;
		res = fread(&tmp, sizeof(float), 1, file);
		std::fclose(file);
		cntR_best_total += tmp;

		file = fopen(_S(ucmDir + iids[i] + "_ev4.bin"), "rb");
		res = fread(&nthresh, sizeof(int), 1, file);
		res = fread(thresh.data(), sizeof(int), nthresh, file);
		vecF tmpRI(nthresh), tmpVOI(nthresh);
		res = fread(tmpRI.data(), sizeof(float), nthresh, file);
		res = fread(tmpVOI.data(), sizeof(float), nthresh, file);
		std::fclose(file);
		for (int t = 0; t < nthresh; t++) {
			globalRI[t] += tmpRI[t];
			globalVOI[t] += tmpVOI[t];
		}

		float tmp1 = -10000, tmp2 = 100000;
		for (int t = 0; t < nthresh; t++) {
			if (tmpRI[t] >= tmp1)
				tmp1 = tmpRI[t];
			if (tmpVOI[t] <= tmp2)
				tmp2 = tmpVOI[t];
		}
		RI_best += tmp1;
		VOI_best += tmp2;
	}

	vecF R(nthresh, 0);
	for (int i = 0; i < nthresh; i++) {
		if (abs(sumR_total[i]) <= DOUBLE_EPS)
			R[i] = cntR_total[i];
		else
			R[i] = cntR_total[i] / sumR_total[i];
	}
	float bestR = -10000;
	int ind = -1;
	for (int i = 0; i < nthresh; i++) {
		if (R[i] > bestR) {
			bestR = R[i];
			ind = i;
		}
	}
	int bestT = thresh[ind];

	file = NULL;
	file = fopen(_S(ucmDir + "eval_cover_img.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fprintf(file, "%10d\n", s_num);
	for (int i = 0; i < s_num; i++) {
		int *ptr_s1 = scores1.ptr<int>(i);
		float *ptr_s2 = scores2.ptr<float>(i);
		fprintf(file, "%10d %10f %10f %10f\n", ptr_s1[0], (float)ptr_s1[1], ptr_s2[0], ptr_s2[1]);
	}
	std::fclose(file);

	file = NULL;
	file = fopen(_S(ucmDir + "eval_cover_th.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fprintf(file, "%10d\n", nthresh);
	for (int i = 0; i < nthresh; i++)
		fprintf(file, "%10f %10f\n", (float)thresh[i], R[i]);
	std::fclose(file);

	float R_best = (abs(sumR_best) <= DOUBLE_EPS) ? cntR_best : (cntR_best / sumR_best);
	float R_best_total = cntR_best_total / sumR_total[0];

	file = NULL;
	file = fopen(_S(ucmDir + "eval_cover.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fprintf(file, "%10f %10f %10f %10f\n", (float)bestT, bestR, R_best, R_best_total);
	std::fclose(file);

	for (int i = 0; i < nthresh; i++) {
		globalRI[i] /= s_num;
		globalVOI[i] /= s_num;
	}
	RI_best /= s_num;
	VOI_best /= s_num;
	float bgRI = -10000, bgVOI = 10000;
	int igRI = -1, igVOI = -1;
	for (int i = 0; i < nthresh; i++) {
		if (globalRI[i] > bgRI) {
			bgRI = globalRI[i];
			igRI = i;
		}
		if (globalVOI[i] < bgVOI) {
			bgVOI = globalVOI[i];
			igVOI = i;
		}
	}

	file = NULL;
	file = fopen(_S(ucmDir + "eval_RI_VOI_thr.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fprintf(file, "%10d\n", nthresh);
	for (int i = 0; i < nthresh; i++)
		fprintf(file, "%10f %10f %10f\n", (float)thresh[i], globalRI[i], globalVOI[i]);
	std::fclose(file);

	file = NULL;
	file = fopen(_S(ucmDir + "eval_RI_VOI.txt"), "w");
	CV_Assert_(file != NULL, "Could not open file for writing.");
	fprintf(file, "%10f %10f %10f %10f %10f %10f\n", (float)thresh[igRI], bgRI, RI_best,
		(float)thresh[igVOI], bgVOI, VOI_best);
	std::fclose(file);
}
