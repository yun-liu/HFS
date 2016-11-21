
#include "../Stdafx.h"
#include "Bench.h"

void plot_eval(CStr evalDir, CStr col)
{
	FILE *file = fopen(_S(evalDir + "result.m"), "w");
	fprintf(file, "clear all;\nclc;\n\n\n");
	fprintf(file, "h = figure;\ntitle('Boundary Benchmark on BSDS');\nhold on;\n\n");
	fprintf(file, "plot(0.700762,0.897659,'go','MarkerFaceColor','g','MarkerEdgeColor','g','MarkerSize',10);\n\n");
	fprintf(file, "[p,r] = meshgrid(0.01:0.01:1,0.01:0.01:1);\n");
	fprintf(file, "F = 2*p.*r./(p+r);\n");
	fprintf(file, "[C,h] = contour(p,r,F);\n\n");
	fprintf(file, "map = zeros(256,3); map(:,1) = 0; map(:,2) = 1; map(:,3) = 0; colormap(map);\n\n");
	fprintf(file, "box on;\ngrid on;\n");
	fprintf(file, "set(gca,'XTick',0:0.1:1);\nset(gca,'YTick',0:0.1:1);\n");
	fprintf(file, "set(gca,'XGrid','on');\nset(gca,'YGrid','on');\n");
	fprintf(file, "xlabel('Recall');\nylabel('Precision');\n");
	fprintf(file, "title('');\naxis square;\naxis([0 1 0 1]);\n\n\n");

	FILE *fid = fopen(_S(evalDir + "eval_bdry_thr.txt"), "r");
	CV_Assert_(fid != NULL, "Can't open file eval_bdry_thr.txt for reading!");
	int nthresh;
	fscanf(fid, "%10d\n", &nthresh);
	vector<Vec4f> prvals(nthresh);
	for (int i = 0; i < nthresh; i++)
		fscanf(fid, "%10f %10f %10f %10f\n", &prvals[i][0], &prvals[i][1], &prvals[i][2], &prvals[i][3]);
	for (int i = 0; i < prvals.size(); i++) {
		if (prvals[i][1] < 0.01)
			prvals.erase(prvals.begin() + i--);
	}
	std::fclose(fid);

	fid = fopen(_S(evalDir + "eval_bdry.txt"), "r");
	CV_Assert_(fid != NULL, "Can't open file eval_bdry.txt for reading!");
	vecF evalRes(8);
	fscanf(fid, "%10f %10f %10f %10f %10f %10f %10f %10f\n", &evalRes[0], &evalRes[1], &evalRes[2],
		&evalRes[3], &evalRes[4], &evalRes[5], &evalRes[6], &evalRes[7]);
	std::fclose(fid);

	fprintf(file, "hold on\nprvals = [");
	const int size_pr = prvals.size();
	for (int i = 0; i < size_pr - 1; i++)
		fprintf(file, "%f %f %f %f;", prvals[i][0], prvals[i][1], prvals[i][2], prvals[i][3]);
	if (size_pr >= 1)
		fprintf(file, "%f %f %f %f", prvals[size_pr - 1][0], prvals[size_pr - 1][1], prvals[size_pr - 1][2], prvals[size_pr - 1][3]);
	fprintf(file, "];\n");
	fprintf(file, "evalRes = [%f %f %f %f %f %f %f %f];\n", evalRes[0], evalRes[1], evalRes[2],
		evalRes[3], evalRes[4], evalRes[5], evalRes[6], evalRes[7]);

	if (size_pr > 1)
		fprintf(file, _S("plot(prvals(1:end,2),prvals(1:end,3),'" + col + "','LineWidth',3);\n"));
	else
		fprintf(file, _S("plot(evalRes(2),evalRes(3),'o','MarkerFaceColor','" + col + "','MarkerEdgeColor','" + col + "','MarkerSize',8);\n"));
	fprintf(file, "hold off\n");

	printf("\nBoundary\n");
	printf("ODS: F( %1.4f, %1.4f ) = %1.4f   [th = %1.2f]\n", evalRes[1], evalRes[2], evalRes[3], evalRes[0]);
	printf("OIS: F( %1.4f, %1.4f ) = %1.4f\n", evalRes[4], evalRes[5], evalRes[6]);
	printf("Area_PR = %1.4f\n", evalRes[7]);

	fid = fopen(_S(evalDir + "eval_cover.txt"), "r");
	CV_Assert_(fid != NULL, "Can't open file eval_cover.txt for reading!");
	vecF evalRegRes(4);
	fscanf(fid, "%10f %10f %10f %10f\n", &evalRegRes[0], &evalRegRes[1], &evalRegRes[2], &evalRegRes[3]);
	std::fclose(fid);
	printf("\nRegion\n");
	printf("GT covering: ODS = %1.2f [th = %1.2f]. OIS = %1.2f. Best = %1.2f\n", evalRegRes[1], evalRegRes[0], evalRegRes[2], evalRegRes[3]);

	fid = fopen(_S(evalDir + "eval_RI_VOI.txt"), "r");
	CV_Assert_(fid != NULL, "Can't open file eval_RI_VOI.txt for reading!");
	vecF evalRegRes1(6);
	fscanf(fid, "%10f %10f %10f %10f %10f %10f\n", &evalRegRes1[0], &evalRegRes1[1], &evalRegRes1[2],
		&evalRegRes1[3], &evalRegRes1[4], &evalRegRes1[5]);
	std::fclose(fid);
	printf("Rand Index: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n", evalRegRes1[1], evalRegRes1[0], evalRegRes1[2]);
	printf("Var. Info.: ODS = %1.2f [th = %1.2f]. OIS = %1.2f.\n", evalRegRes1[4], evalRegRes1[3], evalRegRes1[5]);

	std::fclose(file);
}

void allBench(vecS iids, CStr gtDir, CStr inDir, CStr outDir, int nthresh, float maxDist, bool thinpb)
{
	CmFile::MkDir(outDir);
#pragma omp parallel for
	for (int i = 0; i < iids.size(); i++) {
		CStr evFile1 = outDir + iids[i] + "_ev1.bin";
		CStr evFile2 = outDir + iids[i] + "_ev2.bin";
		CStr evFile3 = outDir + iids[i] + "_ev3.bin";
		CStr evFile4 = outDir + iids[i] + "_ev4.bin";
		if (GetFileAttributesA(_S(evFile4)) != INVALID_FILE_ATTRIBUTES)
			continue;

		CStr inFile = inDir + iids[i];
		CStr b_gtFile = gtDir + "Boundaries/" + iids[i];
		CStr s_gtFile = gtDir + "Segmentation/" + iids[i];

		vecI b_thresh, g_thresh;
		vecF b_cntR, b_sumR, b_cntP, b_sumP, g_cntR, g_sumR, g_cntP, g_sumP;
		float cntR_best;
		evaluation_bdry_image(inFile, b_gtFile, evFile1, b_thresh, b_cntR,
			b_sumR, b_cntP, b_sumP, nthresh, maxDist, thinpb);
		evaluation_reg_image(inFile, s_gtFile, evFile2, evFile3, evFile4, g_thresh,
			g_cntR, g_sumR, g_cntP, g_sumP, &cntR_best, nthresh);
	}
	collect_eval_bdry(outDir);
	collect_eval_reg(outDir);

	vecS ev1, ev2, ev3, ev4;
	CmFile::GetNames(outDir + "*_ev1.bin", ev1);
	CmFile::GetNames(outDir + "*_ev2.bin", ev2);
	CmFile::GetNames(outDir + "*_ev3.bin", ev3);
	CmFile::GetNames(outDir + "*_ev4.bin", ev4);
	CV_Assert_(ev1.size() == ev2.size() && ev2.size() == ev3.size() && ev3.size() == ev4.size(),
		"The sizes of *_ev1.bin,*_ev2.bin,*_ev3.bin and *_ev4.bin don't equal...");
	for (int i = 0; i < ev1.size(); i++) {
		remove(_S(outDir + ev1[i]));
		remove(_S(outDir + ev2[i]));
		remove(_S(outDir + ev3[i]));
		remove(_S(outDir + ev4[i]));
	}
}