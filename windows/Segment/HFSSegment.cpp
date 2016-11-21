#include "Stdafx.h"
#include "HFSSegment.h"
#include "BSDSBench/Bench.h"
#include "Texture.h"
#include "CmIllu.h"

#define MAX_HEIGHT 640
#define MAX_WIDTH  640

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
void print_null(const char *s) {}

HFSSegment::HFSSegment(DataSet &data) :
_bsds(data)
{
	constructEngine();
}

HFSSegment::HFSSegment()
{
	constructEngine();
	float weight1[] = { -0.0024710407f, 0.00608298f, 0.0047505307f, 0.0051097558f, 0.00089799752f };
	float weight2[] = { -0.0040629096f, 0.010430338f, 0.0092625152f, 0.004976281f, 0.0037279273f };
	w1.resize(sizeof(weight1) / sizeof(weight1[0]));
	w2.resize(sizeof(weight2) / sizeof(weight2[0]));
	memcpy(w1.data(), weight1, sizeof(weight1));
	memcpy(w2.data(), weight2, sizeof(weight2));
}

void HFSSegment::constructEngine()
{
	// gSLIC settings
	gSLIC::objects::settings my_settings;
	my_settings.img_size.x = MAX_WIDTH;
	my_settings.img_size.y = MAX_HEIGHT;
	my_settings.spixel_size = 8;
	my_settings.coh_weight = 0.6f;
	my_settings.no_iters = 5;
	my_settings.color_space = gSLIC::CIELAB;    // gSLIC::CIELAB for Lab, or gSLIC::RGB for RGB
	my_settings.seg_method = gSLIC::GIVEN_SIZE; // or gSLIC::DEFAULT_SIZE for default size
	my_settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step

	// Instantiate a core_engine for max width and max height
	gSLIC_engine = new gSLIC::engines::core_engine(my_settings);
	in_img = new gSLIC::UChar4Image(my_settings.img_size, true, true);
	out_img = new gSLIC::UChar4Image(my_settings.img_size, true, true);
}

HFSSegment::~HFSSegment()
{
	delete in_img;
	delete out_img;
	delete gSLIC_engine;
}

void HFSSegment::load_image(const Mat& inimg, gSLIC::UChar4Image* outimg)
{
	gSLIC::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);
	for (int y = 0; y < inimg.rows; y++) {
		const Vec3b *ptr = inimg.ptr<Vec3b>(y);
		for (int x = 0; x < inimg.cols; x++) {
			int idx = x + y * inimg.cols;
			outimg_ptr[idx].b = ptr[x][0];
			outimg_ptr[idx].g = ptr[x][1];
			outimg_ptr[idx].r = ptr[x][2];
		}
	}
}

void HFSSegment::gradientLab(CMat &bgr3u, Mat &mag3u)
{
	Mat lab3u;
	cvtColor(bgr3u, lab3u, COLOR_BGR2Lab);
	const int H = lab3u.rows, W = lab3u.cols;
	Mat Ix(H, W, CV_32SC3), Iy(H, W, CV_32SC3);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<Vec3i>(y, 0) = vecIdtDist(lab3u.at<Vec3b>(y, 1), lab3u.at<Vec3b>(y, 0)) * 2;
		Ix.at<Vec3i>(y, W - 1) = vecIdtDist(lab3u.at<Vec3b>(y, W - 1), lab3u.at<Vec3b>(y, W - 2)) * 2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<Vec3i>(0, x) = vecIdtDist(lab3u.at<Vec3b>(1, x), lab3u.at<Vec3b>(0, x)) * 2;
		Iy.at<Vec3i>(H - 1, x) = vecIdtDist(lab3u.at<Vec3b>(H - 1, x), lab3u.at<Vec3b>(H - 2, x)) * 2;
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++) {
		Vec3i *xP = Ix.ptr<Vec3i>(y);
		const Vec3b *dataP = lab3u.ptr<Vec3b>(y);
		for (int x = 1; x < W - 1; x++)
			xP[x] = vecIdtDist(dataP[x - 1], dataP[x + 1]);
	}
	for (int y = 1; y < H - 1; y++) {
		Vec3i *yP = Iy.ptr<Vec3i>(y);
		const Vec3b *tp = lab3u.ptr<Vec3b>(y - 1);
		const Vec3b *bp = lab3u.ptr<Vec3b>(y + 1);
		for (int x = 0; x < W; x++)
			yP[x] = vecIdtDist(tp[x], bp[x]);
	}

	// Combine x and y direction
	mag3u.create(H, W, CV_8UC3);
	for (int y = 0; y < H; y++) {
		Vec3b *mP = mag3u.ptr<Vec3b>(y);
		Vec3i *xP = Ix.ptr<Vec3i>(y);
		Vec3i *yP = Iy.ptr<Vec3i>(y);
		for (int x = 0; x < W; x++)
			mP[x] = Vec3b(min(xP[x][0] + yP[x][0], 255), min(xP[x][1] + yP[x][1], 255), min(xP[x][2] + yP[x][2], 255));
	}
}

// Get oversegmentation using the GPU version of SLIC
Mat HFSSegment::getSLICIdx(Mat &img3u, int &num_css)
{
	const int _h = img3u.rows;
	const int _w = img3u.cols;
	const int _s = _h*_w;

	load_image(img3u, in_img);
	gSLIC_engine->Set_Image_Size(img3u.cols, img3u.rows);
	gSLIC_engine->Process_Frame(in_img);
	const gSLIC::IntImage *idx_img = gSLIC_engine->Get_Seg_Res();
	int* idx_img_ptr = (int*)idx_img->GetData(MEMORYDEVICE_CPU);

	// get SLIC result and serialize the indexes of the image in idx_img
	num_css = 0;
	int _max = (int)ceil((float)_w / 8.0f)*(int)ceil((float)_h / 8.0f);
	vecI indexes(_max, 0);
	for (int i = 0; i < _s; i++)
		indexes[idx_img_ptr[i]]++;
	for (int i = 0; i < _max; i++)
		indexes[i] = (indexes[i] != 0) ? num_css++ : 0;
	for (int i = 0; i < _s; i++)
		idx_img_ptr[i] = indexes[idx_img_ptr[i]];
	Mat idx_mat(_h, _w, CV_32S, idx_img_ptr);
	idx_mat.convertTo(idx_mat, CV_16U);
	return idx_mat;
}

// Each row of X is a feature vector, with corresponding label in Y. Return a CV_32F weight Mat
Mat HFSSegment::trainSVM(CMat &X1f, const vecF &Y, int sT, double C, double bias, double eps)
{
	// Set SVM parameters
	parameter param; {
		param.solver_type = sT; // L2R_L2LOSS_SVC_DUAL;
		param.C = C;
		param.eps = eps; // see setting below
		param.p = 0.1;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		set_print_string_function(print_null);
		CV_Assert(X1f.rows == Y.size() && X1f.type() == CV_32F);
	}

	// Initialize a problem
	feature_node *x_space = NULL;
	problem prob; {
		prob.l = X1f.rows;
		prob.bias = bias;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(feature_node*, prob.l);
		const int DIM_FEA = X1f.cols;
		prob.n = DIM_FEA + (bias >= 0 ? 1 : 0);
		x_space = Malloc(feature_node, (prob.n + 1) * prob.l);
		int j = 0;
		for (int i = 0; i < prob.l; i++){
			prob.y[i] = Y[i];
			prob.x[i] = &x_space[j];
			const float* xData = X1f.ptr<float>(i);
			for (int k = 0; k < DIM_FEA; k++){
				x_space[j].index = k + 1;
				x_space[j++].value = xData[k];
			}
			if (bias >= 0){
				x_space[j].index = prob.n;
				x_space[j++].value = bias;
			}
			x_space[j++].index = -1;
		}
		CV_Assert(j == (prob.n + 1) * prob.l);
	}

	// Training SVM for current problem
	const char*  error_msg = check_parameter(&prob, &param);
	if (error_msg){
		fprintf(stderr, "ERROR: %s\n", error_msg);
		exit(1);
	}
	model *svmModel = train(&prob, &param);
	Mat wMat(1, prob.n, CV_64F, svmModel->w);
	wMat.convertTo(wMat, CV_32F);
	free_and_destroy_model(&svmModel);
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	return wMat;
}

// X1f is the set of training features, and Y is the set of training labels
Mat HFSSegment::trainSVM(const vecM &X1f, const vecF &Y, int sT, double C, double bias, double eps, int maxTrainNum)
{
	const int numY = Y.size();
	vector<pair<float, int>> score(numY);
	for (int i = 0; i < numY; i++)
		score[i].first = Y[i], score[i].second = i;
	sort(score.begin(), score.end(), std::greater<pair<float, int>>());
	int totalSample = min(numY, maxTrainNum);
	Mat fX1f(totalSample, X1f[0].cols, CV_32F);
	vecF fY(totalSample);
	for (int i = 0; i < totalSample; i++) {
		X1f[score[i].second].copyTo(fX1f.row(i));
		fY[i] = score[i].first;
	}
	return trainSVM(fX1f, fY, sT, C, bias, eps);
}

Vec4f HFSSegment::getColorFeature(Vec3f &in1, Vec3f &in2)
{
	Vec4f feature;
	Vec3f diff = (in1 - in2)/* / 255.0f*/;
	feature[0] = abs(diff[0]), feature[1] = abs(diff[1]);
	feature[2] = abs(diff[2]), feature[3] = getEulerDistance(in1, in2)/* / (255 * SQRT_3)*/;
	return feature;
}

int HFSSegment::getAvgGradientBdry(Mat &idx_mat, vecM &mag1us, Mat &bd_num, vecM &gradients, int num_css)
{
	const int _h = idx_mat.rows;
	const int _w = idx_mat.cols;
	const int size = mag1us.size();

	gradients.resize(size);
	for (int i = 0; i < size; i++) {
		gradients[i].create(num_css, num_css, CV_32F);
		gradients[i] = Scalar::all(0);
	}
	bd_num.create(num_css, num_css, CV_32F);
	bd_num = Scalar::all(0);

	for (int r = 1; r < _h - 1; r++)
	for (int c = 1; c < _w - 1; c++) {
		ushort curr = idx_mat.at<ushort>(r, c), pre, tmp = 0, v[4];
		Point p1(c, r), p2;
		for (int k = 0; k < 4; k++) {
			p2 = p1 + DIRECTION4[k];
			pre = idx_mat.at<ushort>(p2);
			if (pre != curr) {
				bool flag = true;
				for (int t = 0; t < tmp; t++) {
					if (v[t] == pre)
						flag = false;
				}
				if (flag)
					v[tmp++] = pre;
			}
		}

		if (tmp > 0) {
			for (int n = 0; n < size; n++) {
				int u[13]; float m[13];
				for (int k = 0; k < 13; k++) {
					p2 = p1 + CIRCLE2[k];
					if (!CHK_IND(p2)) {
						u[k] = -1, m[k] = 0;
						continue;
					}
					u[k] = idx_mat.at<ushort>(p2);
					m[k] = mag1us[n].at<uchar>(p2);
				}

				for (int t = 0; t < tmp; t++) {
					float m_max = 0;
					for (int k = 0; k < 13; k++) {
						if ((u[k] == curr || u[k] == v[t]) && m[k] > m_max)
							m_max = m[k];
					}
					gradients[n].at<float>(curr, v[t]) += m_max;
					gradients[n].at<float>(v[t], curr) += m_max;
					bd_num.at<float>(curr, v[t])++;
					bd_num.at<float>(v[t], curr)++;
				}
			}
		}
	}

	int num = 0;
	for (int r = 0; r < num_css; r++)
	for (int c = 0; c < num_css; c++) {
		if (abs(bd_num.at<float>(r, c)) > DOUBLE_EPS) {
			for (int i = 0; i < size; i++)
				gradients[i].at<float>(r, c) /= bd_num.at<float>(r, c);
			num++;
		}
	}
	return num;
}

void HFSSegment::getVariance3C(Mat &img3u, Mat &idx_mat, Mat &var, int num_css)
{
	const int _h = img3u.rows, _w = img3u.cols;
	var.create(num_css, num_css, CV_32FC3);
	var = Scalar::all(0);

	vector<Vec3f> sum_color(num_css, Vec3f(0, 0, 0));
	for (int r = 0; r < _h; r++) {
		ushort *ptr1 = idx_mat.ptr<ushort>(r);
		Vec3b *ptr2 = img3u.ptr<Vec3b>(r);
		for (int c = 0; c < _w; c++)
			sum_color[ptr1[c]] += ptr2[c];
	}

	vector<vector<Point>> idx2pix(num_css);
	for (int i = 0; i < num_css; i++)
		idx2pix[i].reserve(_h*_w / num_css);
	for (int r = 0; r < _h; r++) {
		ushort *ptr = idx_mat.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			idx2pix[ptr[c]].push_back(Point(c, r));
	}

	vector<vecI> adjacent(num_css);
	for (int r = 0; r < _h; r++)
	for (int c = 0; c < _w; c++) {
		ushort curr = idx_mat.at<ushort>(r, c), pre;
		for (int k = 0; k < 4; k++) {
			Point p = Point(c, r) + DIRECTION4[k];
			if (CHK_IND(p) && (pre = idx_mat.at<ushort>(p)) != curr
				&& find(adjacent[curr].begin(), adjacent[curr].end(), pre) == adjacent[curr].end())
				adjacent[curr].push_back(pre);
		}
	}

	vector<vector<Vec3f>> avg_color(num_css);
	for (int i = 0; i < num_css; i++){
		avg_color[i].resize(adjacent[i].size());
		for (int j = 0; j < adjacent[i].size(); j++) {
			avg_color[i][j] = (sum_color[i] + sum_color[adjacent[i][j]]) / (float)(idx2pix[i].size() + idx2pix[adjacent[i][j]].size());
		}
	}
	
	for (int i = 0; i < num_css; i++) {
		Vec3f *ptr = var.ptr<Vec3f>(i);
		for (int j = 0; j < adjacent[i].size(); j++) {
			int temp = adjacent[i][j];
			Vec3f mean = avg_color[i][j];
			for (int k = 0; k < idx2pix[i].size(); k++) {
				Vec3b curr = img3u.at<Vec3b>(idx2pix[i][k]);
				ptr[temp] += Vec3f(pow(curr[0] - mean[0], 2), pow(curr[1] - mean[1], 2), pow(curr[2] - mean[2], 2));
			}
			for (int k = 0; k < idx2pix[temp].size(); k++) {
				Vec3b curr = img3u.at<Vec3b>(idx2pix[temp][k]);
				ptr[temp] += Vec3f(pow(curr[0] - mean[0], 2), pow(curr[1] - mean[1], 2), pow(curr[2] - mean[2], 2));
			}
			ptr[temp] /= (float)(idx2pix[i].size() + idx2pix[temp].size());
		}
	}
}

void HFSSegment::getHistColor(Mat &img3u, Mat &idx_mat, Mat &hist, int num_css)
{
	const int _h = img3u.rows, _w = img3u.cols;
	const int nbins = 8, nrage = 256 / 8;
	hist.create(num_css, nbins * nbins * nbins, CV_32F);
	hist = Scalar::all(0);
	for (int r = 0; r < _h; r++) {
		Vec3b *ptr1 = img3u.ptr<Vec3b>(r);
		ushort *ptr2 = idx_mat.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			hist.at<float>(ptr2[c], ptr1[c][0] / nrage*nbins*nbins + ptr1[c][1] / nrage*nbins + ptr1[c][2] / nrage)++;
	}
	for (int i = 0; i < num_css; i++) {
		int s = (int)sum(hist.row(i))[0];
		if (s != 0)
			hist.row(i) /= s;
	}
}

void HFSSegment::getHistGradient(Mat &img3u, Mat &idx_mat, Mat &hist, int num_css)
{
	const int _h = img3u.rows, _w = img3u.cols;
	Mat Ix(_h, _w, CV_32F), Iy(_h, _w, CV_32F), gray;
	cvtColor(img3u, gray, COLOR_BGR2GRAY);

	for (int r = 0; r < _h; r++) {
		Ix.at<float>(r, 0) = (float)((gray.at<uchar>(r, 1) - gray.at<uchar>(r, 0)) * 2);
		Ix.at<float>(r, _w - 1) = (float)((gray.at<uchar>(r, _w - 1) - gray.at<uchar>(r, _w - 2)) * 2);
	}
	for (int c = 0; c < _w; c++) {
		Iy.at<float>(0, c) = (float)((gray.at<uchar>(1, c) - gray.at<uchar>(0, c)) * 2);
		Iy.at<float>(_h - 1, c) = (float)((gray.at<uchar>(_h - 1, c) - gray.at<uchar>(_h - 2, c)) * 2);
	}

	for (int r = 0; r < _h; r++) {
		uchar *ptr1 = gray.ptr<uchar>(r);
		float *ptr2 = Ix.ptr<float>(r);
		for (int c = 1; c < _w - 1; c++)
			ptr2[c] = (float)(ptr1[c + 1] - ptr1[c - 1]);
	}
	for (int r = 1; r < _h - 1; r++) {
		uchar *ptr1 = gray.ptr<uchar>(r - 1);
		uchar *ptr2 = gray.ptr<uchar>(r + 1);
		float *ptr3 = Iy.ptr<float>(r);
		for (int c = 0; c < _w; c++)
			ptr3[c] = (float)(ptr2[c] - ptr1[c]);
	}

	Mat mag(_h, _w, CV_32F), angle(_h, _w, CV_32F);
	for (int r = 0; r < _h; r++) {
		float *ptr1 = mag.ptr<float>(r);
		float *ptr2 = angle.ptr<float>(r);
		float *ptr3 = Ix.ptr<float>(r);
		float *ptr4 = Iy.ptr<float>(r);
		for (int c = 0; c < _w; c++) {
			ptr1[c] = abs(ptr3[c]) + abs(ptr4[c]);
			float t = atan2(ptr4[c], ptr3[c]);
			t = t >= 0 ? t : (t + PI);
			ptr2[c] = (abs(t - PI) < DOUBLE_EPS) ? 0 : t;
		}
	}

	float nrage = PI / 9;
	hist.create(num_css, 9, CV_32F);
	hist = Scalar::all(0);
	for (int r = 0; r < _h; r++) {
		float *ptr1 = mag.ptr<float>(r);
		float *ptr2 = angle.ptr<float>(r);
		ushort *ptr3 = idx_mat.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			hist.at<float>(ptr3[c], (int)floor(ptr2[c] / nrage)) += ptr1[c];
	}

	for (int i = 0; i < num_css; i++)
		hist.row(i) /= sum(hist.row(i))[0];
}

float HFSSegment::getChiHist(Mat &hist1, Mat &hist2)
{
	CV_Assert_(hist1.cols == hist2.cols, "The Column of histgrams must be equal...");
	float chi = 0;
	for (int i = 0; i < hist1.cols; i++) {
		float v1 = hist1.at<float>(0, i), v2 = hist2.at<float>(0, i);
		float _sum = v1 + v2, _sub = v1 - v2;
		chi += (abs(_sum)>DOUBLE_EPS) ? (_sub*_sub) / _sum : 0;
	}
	return (chi/**0.5f*/);
}

void HFSSegment::generateTrainData()
{
	const int nbins = 8;
	const int nrage = 256 / nbins;

	const int trainNum = _bsds.trainNum;
	vecM idxes(trainNum), hists(trainNum), flags(trainNum), gtValues(trainNum);

	string iluDir = _bsds.ResDir + "Ilu/";
	CmFile::MkDir(iluDir);

	for (int n = 0; n < trainNum; n++) {
		Mat img3u, mag3u;
		img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.trainSet[n])));
		gradientLab(img3u, mag3u);
		const int _h = img3u.rows, _w = img3u.cols, _s = _h*_w;

		// get SLIC map and serialize the indexes of the map
		int numInd;
		Mat idx_mat = getSLICIdx(img3u, numInd);
		idx_mat.copyTo(idxes[n]);

		// get histogram of every superpixels
		vecI numSP(numInd, 0);
		Mat hist(numInd, 3 * nbins, CV_32F, Scalar(0));
		for (int r = 0; r < _h; r++) {
			Vec3b *mP = mag3u.ptr<Vec3b>(r);
			for (int c = 0; c < _w; c++) {
				ushort tmp = idx_mat.at<ushort>(r, c);
				hist.at<float>(tmp, mP[c][0] / nrage + nbins * 0)++;
				hist.at<float>(tmp, mP[c][1] / nrage + nbins * 1)++;
				hist.at<float>(tmp, mP[c][2] / nrage + nbins * 2)++;
				numSP[tmp]++;
			}
		}
		for (int i = 0; i < numInd; i++)
			hist.row(i) /= numSP[i];
		hist.copyTo(hists[n]);

		vecM segs;
		matRead(_bsds.gtSegsDir + _bsds.trainSet[n], segs);

		////* Illustrate corresponding data 
		//{
		//	imwrite(iluDir + _bsds.trainSet[n] + ".jpg", img3u);
		//	imwrite(iluDir + _bsds.trainSet[n] + "_LabG.png", mag3u);
		//	for (uint i = 0; i < segs.size(); i++){
		//		Mat_<ushort> matIlu = segs[i];
		//		CmIllu::saveRgbLabel(iluDir + _bsds.trainSet[n] + format("_Seg%d.png", i), matIlu);
		//	}
		//}//*/

		const int SEG_NUM = segs.size();
		Mat flag(SEG_NUM, numInd, CV_32S, Scalar(0));
		Mat gtValue(SEG_NUM, numInd, CV_32S);
		for (int i = 0; i < SEG_NUM; i++) {
			vector<map<int, int>> maplive(numInd);
			for (int r = 0; r < _h; r++) {
				ushort* ptr1 = idx_mat.ptr<ushort>(r);
				ushort* ptr2 = segs[i].ptr<ushort>(r);
				for (int c = 0; c < _w; c++)
					maplive[ptr1[c]][ptr2[c]]++;
			}
			for (int j = 0; j < numInd; j++) {
				int label = 0, max = 0, area = 0;
				for (map<int, int>::iterator it = maplive[j].begin(); it != maplive[j].end(); area += (it++)->second)
				if (it->second > max) label = it->first, max = it->second;
				if ((double)max / area >= 0.9)
					flag.at<int>(i, j) = 1;
				gtValue.at<int>(i, j) = label;
			}
		}
		flag.copyTo(flags[n]);
		gtValue.copyTo(gtValues[n]);
	}
	matWrite(_bsds.ResDir + "Segment.idx", idxes);
	matWrite(_bsds.ResDir + "Segment.hit", hists);
	matWrite(_bsds.ResDir + "Segment.flg", flags);
	matWrite(_bsds.ResDir + "Segment.r2v", gtValues);
}

// Genarate training data for the second training stage, and we have provided the model.
// If you want to run it by yourself, it will take you very very long time.
// Thus, be careful please!
void HFSSegment::mergePartial()
{
	const int trainNum = _bsds.trainNum;

	loadTrainedModel(_bsds.ResDir + "Segment.WS");
	vecI num_css(trainNum);
	vecM idx_mats(trainNum), segs(trainNum), relations(trainNum);

	for (int i = 0; i < trainNum; i++) {
		Mat img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.trainSet[i])));
		Mat idx_mat = getSLICIdx(img3u, num_css[i]);
		idx_mat.copyTo(idx_mats[i]);
	}

#pragma omp parallel for
	for (int n = 0; n < trainNum; n++) {
		printf("%d  ", n);
		Mat &seg = segs[n], &relation = relations[n];
		Mat img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.trainSet[n]))), lab3u, mag1u;
		cv::cvtColor(img3u, lab3u, CV_BGR2Lab);
		mag_engine.process_img(img3u, mag1u);
		getSegmentationI(seg, lab3u, mag1u, idx_mats[n], num_css[n], 0.1f, 0);

		int _h = img3u.rows;
		int _w = img3u.cols;
		relation.create(num_css[n], num_css[n], CV_8U);
		relation = Scalar::all(0);

		// get adjacent domains
		vector<vector<ushort>> adjacent(num_css[n]);
		for (int r = 1; r < _h - 1; r++)
		for (int c = 1; c < _w - 1; c++) {
			ushort curr = seg.at<ushort>(r, c);
			for (int k = 0; k < 4; k++) {
				ushort pre = seg.at<ushort>(Point(c, r) + DIRECTION4[k]);
				if (curr > pre&&find(adjacent[curr].begin(), adjacent[curr].end(), pre) == adjacent[curr].end())
					adjacent[curr].push_back(pre);
			}
		}

		vecM groundTruth;
		matRead(_bsds.gtBdryDir + _bsds.trainSet[n], groundTruth);

		float f1 = evaluation_bdry_single_image(seg, groundTruth), f2;

		Mat tmp_idx;
		for (int i = 0; i < num_css[n]; i++)
		for (int j = 0; j < adjacent[i].size(); j++) {
			seg.copyTo(tmp_idx);
			ushort v = adjacent[i][j];
			for (int r = 0; r < _h; r++) {
				ushort *data = tmp_idx.ptr<ushort>(r);
				for (int c = 0; c < _w; c++) {
					if (data[c] == i)
						data[c] = v;
				}
			}

			f2 = evaluation_bdry_single_image(tmp_idx, groundTruth);
			if (f1 < f2)
				relation.at<uchar>(i, v) = relation.at<uchar>(v, i) = 2;
			else
				relation.at<uchar>(i, v) = relation.at<uchar>(v, i) = 1;
		}
	}

	matWrite(_bsds.ResDir + "mergePartial.idx", segs);
	matWrite(_bsds.ResDir + "mergePartial.rlt", relations);
}

void HFSSegment::trainStageI(int trainSample)
{
	const int trainNum = _bsds.trainNum;
	vecM X1f;             vecF Y;
	X1f.reserve(500000);  Y.reserve(500000);

	vecM idxes, hists, flags, gtValues;
	matRead(_bsds.ResDir + "Segment.idx", idxes);
	matRead(_bsds.ResDir + "Segment.hit", hists);
	matRead(_bsds.ResDir + "Segment.flg", flags);
	matRead(_bsds.ResDir + "Segment.r2v", gtValues);
	CV_Assert_(idxes.size() == trainNum&&hists.size() == trainNum, "Load data error...");
	CV_Assert_(flags.size() == trainNum&&gtValues.size() == trainNum, "Load data error...");

	for (int n = 0; n < trainNum; n++) {
		const int _h = idxes[n].rows, _w = idxes[n].cols;
		const int num_css = flags[n].cols, num_seg = flags[n].rows;
		Mat &idx = idxes[n], &hist = hists[n], &flag = flags[n], &gtValue = gtValues[n];

		Mat img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.trainSet[n]))), lab3u, mag1u;
		cv::cvtColor(img3u, lab3u, CV_BGR2Lab);
		mag_engine.process_img(img3u, mag1u);

		// get adjacent domains
		vector<vecI> adjacent(num_css), bdPixNum(num_css);
		vector<vecF> bdGradient(num_css);
		for (int r = 1; r < _h - 1; r++)
		for (int c = 1; c < _w - 1; c++) {
			ushort curr = idx.at<ushort>(r, c);
			for (int k = 0; k < 4; k++) {
				Point p = Point(c, r) + DIRECTION4[k];
				ushort pre = idx.at<ushort>(p);
				if (curr > pre) {
					float maxG = max(mag1u.at<uchar>(p), mag1u.at<uchar>(r, c));
					vecI::iterator iter = find(adjacent[curr].begin(), adjacent[curr].end(), pre);
					if (iter == adjacent[curr].end()) {
						adjacent[curr].push_back(pre);
						bdGradient[curr].push_back(maxG);
						bdPixNum[curr].push_back(1);
					}
					else {
						int temp = (int)(iter - adjacent[curr].begin());
						bdGradient[curr][temp] += maxG;
						bdPixNum[curr][temp] += 1;
					}
				}
			}
		}
		for (int i = 0; i < num_css; i++)
		for (int j = 0; j < adjacent[i].size(); j++)
			bdGradient[i][j] /= bdPixNum[i][j];

		vector<Vec3f> avgColor(num_css, Vec3f(0, 0, 0));
		vecI numR(num_css, 0);
		for (int r = 0; r < _h; r++) {
			ushort *iP = idx.ptr<ushort>(r);
			Vec3b *cP = lab3u.ptr<Vec3b>(r);
			for (int c = 0; c < _w; c++)
				avgColor[iP[c]] += cP[c], numR[iP[c]]++;
		}
		for (int i = 0; i < num_css; i++)
			avgColor[i] /= numR[i];

		vector<pair<float, pair<int, int>>> simuRes;
		simuRes.reserve(5000);
		for (int i = 0; i < num_css; i++) {
			for (int j = 0; j < adjacent[i].size(); j++) {
				float score = getChiHist(hist.row(i), hist.row(adjacent[i][j]));
				simuRes.push_back(make_pair(score, make_pair(i, adjacent[i][j])));
			}
		}
		sort(simuRes.begin(), simuRes.end(), std::greater<pair<float, pair<int, int>>>());

		/*Mat hist1, hist2, var1, var2;
		getHistColor(img3u, idx, hist1, num_css);
		getHistGradient(img3u, idx, hist2, num_css);
		getVariance3C(img3u, idx, var1, num_css);
		getVariance3C(lab3u, idx, var2, num_css);*/
		
		int num = min(trainSample, (int)simuRes.size());
		for (int m = 0; m < num; m++) {
			int v1 = simuRes[m].second.first, v2 = simuRes[m].second.second;
			//if (flag.at<int>(0, v1) || flag.at<int>(0, v2)) continue;
			Mat score(1, 5, CV_32F);
			Vec4f fcolor = getColorFeature(avgColor[v1], avgColor[v2]);
			score.at<float>(0, 0) = fcolor[0], score.at<float>(0, 1) = fcolor[1];
			score.at<float>(0, 2) = fcolor[2], score.at<float>(0, 3) = fcolor[3];

			int v1t = v1 > v2 ? v1 : v2, v2t = v1 > v2 ? v2 : v1;
			for (int i = 0; i < adjacent[v1t].size(); i++) {
				if (adjacent[v1t][i] == v2t)
					score.at<float>(0, 4) = bdGradient[v1t][i];
			}

			/*score.at<float>(0, 5) = getChiHist(hist1.row(v1), hist1.row(v2)) * 255;
			score.at<float>(0, 6) = getChiHist(hist2.row(v1), hist2.row(v2)) * 255;
			for (int i = 0; i < 3; i++)
				score.at<float>(0, 7 + i) = var1.at<Vec3f>(v1, v2)[i];
			for (int i = 0; i < 3; i++)
				score.at<float>(0, 10 + i) = var2.at<Vec3f>(v1, v2)[i];*/

			float s = 0;
			for (int k = 0; k < num_seg; k++) {
				if (flag.at<int>(k, v1) && flag.at<int>(k, v2))
					s += (gtValue.at<int>(k, v1) != gtValue.at<int>(k, v2)) ? 1 : 0;
			}
			X1f.push_back(score);
			Y.push_back(s / num_seg);
		}
	}

	Mat crntM = trainSVM(X1f, Y, L2R_L2LOSS_SVR, 100, 1);
	wMat.push_back(crntM.colRange(0, crntM.cols - 1));
	CV_Assert(wMat[0].rows == 1 && wMat[0].cols == 5);
	matWrite(_bsds.ResDir + "Segment.WS", wMat);
}

void HFSSegment::trainStageII()
{
	const int trainNum = _bsds.trainNum;
	vecM segs(trainNum), relations(trainNum);
	matRead(_bsds.ResDir + "mergePartial.idx", segs);
	matRead(_bsds.ResDir + "mergePartial.rlt", relations);

	//vecM filters;
	//Texture::texture_filters(filters, 8, 1.4);

	vecM X1f;             vecF Y;
	X1f.reserve(120000);  Y.reserve(120000);

	for (int n = 0; n < trainNum; n++) {
		Mat &idx_mat = segs[n], &relation = relations[n];
		const int _h = idx_mat.rows;
		const int _w = idx_mat.cols;
		const int num_css = relation.rows;

		Mat img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.trainSet[n]))), lab3u, mag1u;
		cv::cvtColor(img3u, lab3u, CV_BGR2Lab);
		mag_engine.process_img(img3u, mag1u);

		vecM mag1us, gradients;
		Mat bd_num, texture;
		//Texture::texture_gradient(img3u, filters, texture);
		mag1us.push_back(mag1u);
		//mag1us.push_back(texture);
		getAvgGradientBdry(idx_mat, mag1us, bd_num, gradients, num_css);
		const int size = gradients.size();

		vecI num_pix(num_css, 0);
		vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
		for (int r = 0; r < _h; r++) {
			ushort *idx_ptr = idx_mat.ptr<ushort>(r);
			Vec3b *clr_ptr = lab3u.ptr<Vec3b>(r);
			for (int c = 0; c < _w; c++)
				num_pix[idx_ptr[c]]++, avg_color[idx_ptr[c]] += clr_ptr[c];
		}
		for (int i = 1; i < num_css; i++)
			avg_color[i] /= num_pix[i];
        
		/*Mat hist1, hist2, var1, var2;
		getHistColor(img3u, idx_mat, hist1, num_css);
		getHistGradient(img3u, idx_mat, hist2, num_css);
		getVariance3C(img3u, idx_mat, var1, num_css);
		getVariance3C(lab3u, idx_mat, var2, num_css);*/

		for (int r = 0; r < num_css; r++)
		for (int c = 0; c < r; c++) {
			if (relation.at<uchar>(r, c) == 0 || num_pix[r] < 100 || num_pix[c] < 100)
				continue;
			Mat score(1, 5, CV_32F);
			Vec4f fcolor = getColorFeature(avg_color[r], avg_color[c]);
			score.at<float>(0, 0) = fcolor[0], score.at<float>(0, 1) = fcolor[1];
			score.at<float>(0, 2) = fcolor[2], score.at<float>(0, 3) = fcolor[3];
			score.at<float>(0, 4) = gradients[0].at<float>(r, c);
			//for (int i = 0; i < size; i++)
				//score.at<float>(0, 4 + i) = gradients[i].at<float>(r, c);

			/*score.at<float>(0, 5) = getChiHist(hist1.row(r), hist1.row(c)) * 256;
			score.at<float>(0, 6) = getChiHist(hist2.row(r), hist2.row(c)) * 256;
			for (int i = 0; i < 3; i++)
				score.at<float>(0, 7 + i) = var1.at<Vec3f>(r, c)[i];
			for (int i = 0; i < 3; i++)
				score.at<float>(0, 10 + i) = var2.at<Vec3f>(r, c)[i];*/

			/*if (relation.at<uchar>(r, c) == 1)
				pX1f.push_back(score);
			else
				nX1f.push_back(score);*/
			X1f.push_back(score);
			Y.push_back(relation.at<uchar>(r, c) == 1);
		}
	}

	Mat crntM = trainSVM(X1f, Y, L1R_L2LOSS_SVC, 100, 1);
	wMat.push_back(crntM.colRange(0, crntM.cols - 1));
	CV_Assert(wMat[1].rows == 1 && wMat[1].cols == 5);
	matWrite(_bsds.ResDir + "Segment.WS", wMat);
}

void HFSSegment::trainSegment()
{
	StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);

	sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
	generateTrainData();
	sdkStopTimer(&my_timer);
	std::cout << endl << "generate training data in:[" << sdkGetTimerValue(&my_timer) / 1000 << "]s" << endl;
	
	sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
	trainStageI();
	sdkStopTimer(&my_timer);
	std::cout << endl << "train stage I in:[" << sdkGetTimerValue(&my_timer) / 1000 << "]s" << endl;

	sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
	//mergePartial();
	trainStageII();
	sdkStopTimer(&my_timer);
	std::cout << endl << "train stage II in:[" << sdkGetTimerValue(&my_timer) / 1000 << "]s" << endl;
	cout << endl;
}

// Load trained data
void HFSSegment::loadTrainedModel(CStr path)
{
	if (CmFile::FileExist(path)) {
		wMat.clear();
		matRead(path, wMat);
		w1.resize(wMat[0].cols);
		for (int i = 0; i < wMat[0].cols; i++)
			w1[i] = wMat[0].at<float>(0, i);
		if (wMat.size() < 2)
			return;
		w2.resize(wMat[1].cols);
		for (int i = 0; i < wMat[1].cols; i++)
			w2[i] = wMat[1].at<float>(0, i);
	}
	else {
		cout << "Can not find the trained model Segment.WS!" << endl;
		exit(1);
	}
}

void HFSSegment::getSegmentationI(Mat &seg, Mat &lab3u, Mat &mag1u, Mat &idx_mat, int &num_css, float c, int min_size)
{
	const int _h = lab3u.rows;
	const int _w = lab3u.cols;
	const int _s = _h*_w;

	// get adjacent domains
	vector<vecI> adjacent(num_css), bdPixNum(num_css);
	vector<vecF> bdGradient(num_css);
	for (int r = 1; r < _h - 1; r++)
	for (int c = 1; c < _w - 1; c++) {
		ushort curr = idx_mat.at<ushort>(r, c);
		for (int k = 0; k < 4; k++) {
			Point p = Point(c, r) + DIRECTION4[k];
			ushort pre = idx_mat.at<ushort>(p);
			if (curr > pre) {
				float maxG = max(mag1u.at<uchar>(p), mag1u.at<uchar>(r, c));
				vecI::iterator iter = find(adjacent[curr].begin(), adjacent[curr].end(), pre);
				if (iter == adjacent[curr].end()) {
					adjacent[curr].push_back(pre);
					bdGradient[curr].push_back(maxG);
					bdPixNum[curr].push_back(1);
				}
				else {
					int temp = (int)(iter - adjacent[curr].begin());
					bdGradient[curr][temp] += maxG;
					bdPixNum[curr][temp] += 1;
				}
			}
		}
	}
	for (int i = 0; i < num_css; i++)
	for (int j = 0; j < adjacent[i].size(); j++)
		bdGradient[i][j] /= bdPixNum[i][j];

	int num = 0;
	for (int i = 0; i < num_css; i++)
		num += adjacent[i].size();

	// get features of every superpixel
	vecI numR(num_css, 0);
	vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
	for (int r = 0; r < _h; r++) {
		ushort *iP = idx_mat.ptr<ushort>(r);
		Vec3b *cP = lab3u.ptr<Vec3b>(r);
		for (int c = 0; c < _w; c++)
			avg_color[iP[c]] += cP[c], numR[iP[c]]++;
	}
	for (int i = 0; i < num_css; i++)
		avg_color[i] /= numR[i];
	
	// get the edges
	edge *edges = new edge[num];
	int index = 0;
	for (int i = 0; i < num_css; i++) {
		int adjaNum = adjacent[i].size();
		for (int j = 0; j < adjaNum; j++) {
			edges[index].a = i;
			edges[index].b = adjacent[i][j];
			Vec4f fcolor = getColorFeature(avg_color[i], avg_color[adjacent[i][j]]);
			edges[index++].w = fcolor[0] * w1[0] + fcolor[1] * w1[1] + fcolor[2] * w1[2] + fcolor[3] * w1[3] + bdGradient[i][j] * w1[4];
		}
	}
	CV_Assert(num == index);

	// segment
	universe *u = segment_graph(num_css, num, edges, c, numR);
	// post process small components
	for (int i = 0; i < num; i++) {
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size1(a) < min_size) || (u->size1(b) < min_size)))
			u->join(a, b);
	}

	// compute the indexes of superpixels in seg
	int idx = 1; vecI reg2ind(num_css), indexes(num_css);
	std::memset(indexes.data(), 0, num_css*sizeof(int));
	for (int i = 0; i < num_css; i++) {
		int comp = u->find(i);
		if (!indexes[comp])
			indexes[comp] = idx++;
		reg2ind[i] = indexes[comp];
	}
	CV_Assert(u->num_sets() == idx - 1);
	seg.create(_h, _w, CV_16U);
	for (int r = 0; r < _h; r++) {
		ushort *sP = seg.ptr<ushort>(r);
		ushort *iP = idx_mat.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			sP[c] = reg2ind[iP[c]];
	}
	delete u;
	delete[] edges;

	num_css = idx;
}

void HFSSegment::getSegmentationII(Mat &seg, Mat &img3u, Mat &lab3u, Mat &mag1u, Mat &idx_mat, int &num_css, float c, int min_size)
{
	const int _h = lab3u.rows;
	const int _w = lab3u.cols;
	const int _s = _h*_w;

	//vecM filters;
	//Texture::texture_filters(filters, 8, 1.4);

	vecM mag1us, gradients;
	Mat bd_num, texture;
	//Texture::texture_gradient(img3u, filters, texture);
	mag1us.push_back(mag1u);
	//mag1us.push_back(texture);
	int num = getAvgGradientBdry(idx_mat, mag1us, bd_num, gradients, num_css);
	const int size = gradients.size();
	CV_Assert(num % 2 == 0);
	num /= 2;

	vecI num_pix(num_css, 0);
	vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
	for (int r = 0; r < _h; r++) {
		ushort *idx_ptr = idx_mat.ptr<ushort>(r);
		Vec3b *clr_ptr = lab3u.ptr<Vec3b>(r);
		for (int c = 0; c < _w; c++)
			num_pix[idx_ptr[c]]++, avg_color[idx_ptr[c]] += clr_ptr[c];
	}
	for (int i = 1; i < num_css; i++)
		avg_color[i] /= num_pix[i];

	edge *edges = new edge[num];
	int index = 0;
	for (int r = 0; r < num_css; r++)
	for (int c = 0; c < r; c++) {
		if (bd_num.at<int>(r, c) == 0) continue;
		edges[index].a = r;
		edges[index].b = c;
		Vec4f fcolor = getColorFeature(avg_color[r], avg_color[c]);
		edges[index].w = fcolor[0] * w2[0] + fcolor[1] * w2[1] + fcolor[2] * w2[2] + fcolor[3] * w2[3];
		//for (int i = 0; i < size; i++)
			//edges[index].w += gradients[i].at<float>(r, c)*w2[i + 4];
		edges[index].w += gradients[0].at<float>(r, c)*w2[4];
		index++;
	}
	CV_Assert(num == index);

	// segment
	universe *u = segment_graph(num_css, num, edges, c, num_pix);
	// post process small components
	for (int i = 0; i < num; i++) {
		int a = u->find(edges[i].a);
		int b = u->find(edges[i].b);
		if ((a != b) && ((u->size1(a) < min_size) || (u->size1(b) < min_size)))
			u->join(a, b);
	}

	// compute the indexes of superpixels in seg
	int idx = 1;
	vecI reg2ind(num_css), indexes(num_css, 0);
	for (int i = 1; i < num_css; i++) {
		int comp = u->find(i);
		if (!indexes[comp])
			indexes[comp] = idx++;
		reg2ind[i] = indexes[comp];
	}
	CV_Assert(u->num_sets() == idx);
	seg.create(_h, _w, CV_16U);
	for (int r = 0; r < _h; r++) {
		ushort *sP = seg.ptr<ushort>(r);
		ushort *iP = idx_mat.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			sP[c] = reg2ind[iP[c]];
	}
	delete u;
	delete[] edges;

	num_css = idx - 1;
}

void HFSSegment::getSegmentationIII(Mat &seg, Mat &hed, Mat &idx_mat, int &num_css, float threshold)
{
	const int _h = idx_mat.rows;
	const int _w = idx_mat.cols;
	const int _ds = num_css + 1;

	Mat total_num(_ds, _ds, CV_32S, Scalar(0));
	Mat edge_val(_ds, _ds, CV_32S, Scalar(0));
	
	for (int r = 0; r < _h; r++)
	for (int c = 0; c < _w; c++) {
		ushort curr = idx_mat.at<ushort>(r, c), pre;
		for (int k = 0; k < 4; k++) {
			Point p = Point(c, r) + DIRECTION4[k];
			if (CHK_IND(p) && (pre = idx_mat.at<ushort>(p)) != curr) {
				total_num.at<int>(curr, pre)++, total_num.at<int>(pre, curr)++;
				int max = 0;
				for (int t1 = -4; t1 <= 4; t1++)
				for (int t2 = -4; t2 <= 4; t2++) {
					if (t1*t1 + t2*t2 > 12) continue;
					p = Point(c, r) + Point(t1, t2);
					if (CHK_IND(p) && hed.at<Vec3b>(p)[0] > max)
						max = hed.at<Vec3b>(p)[0];
				}
				edge_val.at<int>(curr, pre) += max, edge_val.at<int>(pre, curr) += max;
			}
		}
	}
	
	universe *u = new universe(_ds);
	for (int r = 1; r < _ds; r++)
	for (int c = 1; c < r; c++) {
		if (total_num.at<int>(r, c) > 0 && (float)edge_val.at<int>(r, c) / (float)total_num.at<int>(r, c) < threshold) {
			int a = u->find(r);
			int b = u->find(c);
			if (a != b)  u->join(a, b);
		}
	}
	int idx = 1;
	vecI reg2ind(_ds), indexes(_ds, 0);
	for (int i = 1; i < _ds; i++) {
		int comp = u->find(i);
		if (!indexes[comp])
			indexes[comp] = idx++;
		reg2ind[i] = indexes[comp];
	}
	CV_Assert(u->num_sets() == idx);
	num_css = idx - 1;

	seg.create(_h, _w, CV_16U);
	for (int r = 0; r < _h; r++) {
		ushort *ptr1 = idx_mat.ptr<ushort>(r);
		ushort *ptr2 = seg.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			ptr2[c] = reg2ind[ptr1[c]];
	}
}

void HFSSegment::drawSegmentationRes(Mat &show, Mat &seg, Mat &img3u, int num_css)
{
	const int _h = img3u.rows;
	const int _w = img3u.cols;

	// Get the average color of every region
	vecI region_size(num_css, 0);
	vector<Vec3f> avg_color(num_css, Vec3f(0, 0, 0));
	for (int r = 0; r < _h; r++) {
		Vec3b *imP = img3u.ptr<Vec3b>(r);
		ushort *segP = seg.ptr<ushort>(r);
		for (int c = 0; c < _w; c++) {
			avg_color[segP[c] - 1] += imP[c];
			region_size[segP[c] - 1]++;
		}
	}
	for (int i = 0; i < num_css; i++)
		avg_color[i] /= region_size[i];

	// Construct the display image of segmentation
	show.create(img3u.size(), img3u.type());
	for (int r = 0; r < _h; r++) {
		Vec3b *data = show.ptr<Vec3b>(r);
		ushort *seg_ptr = seg.ptr<ushort>(r);
		for (int c = 0; c < _w; c++)
			data[c] = avg_color[seg_ptr[c] - 1];
	}
}

// Run BSDS500 dataset
void HFSSegment::runDataSet(float c, int min_size)
{
	trainSegment();
	loadTrainedModel(_bsds.ResDir + "Segment.WS");
	
	const int testNum = _bsds.testNum;
	Mat img3u, idx_mat, lab3u, mag1u, tmp;  
	vecI num_css(testNum);
	vecM segs(testNum), shows(testNum), segs0(testNum);
	
	StopWatchInterface *my_timer; sdkCreateTimer(&my_timer);
	sdkResetTimer(&my_timer); sdkStartTimer(&my_timer);
	for (int i = 0; i < testNum; i++) {
		//printf("\r%06d", i);
		img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.testSet[i])));
		//GaussianBlur(img3u, img3u, Size(7, 7), 1, 1);
		idx_mat = getSLICIdx(img3u, num_css[i]);
		cv::cvtColor(img3u, lab3u, CV_BGR2Lab);
		mag_engine.process_img(img3u, mag1u);

		getSegmentationI(tmp, lab3u, mag1u, idx_mat, num_css[i], 0.08f, 100);
		getSegmentationII(segs[i], img3u, lab3u, mag1u, tmp, num_css[i], c, min_size); // 0.1f, 150);
		//Mat hed = imread(_bsds.WkDir + format("HED/PNG/%d.png", atoi(_S(_bsds.testSet[i]))));
		//getSegmentationIII(segs[i], hed, segs0[i], num_css[i], 70.0f);
	}
	sdkStopTimer(&my_timer);
	std::cout << endl << "segmentation average in:[" << sdkGetTimerValue(&my_timer) / testNum << "]ms" << endl;

	// Write the map of segmentation and the picture for displaying to disk
	CStr _FResDir = _bsds.ResDir + format("Segmentation/");
	CStr _FDisDir = _bsds.ResDir + format("SegDisplay/");
	if (GetFileAttributesA(_S(_FResDir)) == INVALID_FILE_ATTRIBUTES)
		CmFile::MkDir(_FResDir);
	if (GetFileAttributesA(_S(_FDisDir)) == INVALID_FILE_ATTRIBUTES)
		CmFile::MkDir(_FDisDir);
	for (int i = 0; i < testNum; i++) {
		vecM tmp(1, segs[i]);
		matWrite(_FResDir + _bsds.testSet[i], tmp);
		img3u = imread(format(_S(_bsds.ImgDir), _S(_bsds.testSet[i])));
		drawSegmentationRes(shows[i], segs[i], img3u, num_css[i]);
		imwrite(_FDisDir + _bsds.testSet[i] + format("_HFS.png"), shows[i]);
	}

	allBench(_bsds.testSet, _bsds.gtDir, _FResDir, _FResDir + "eval/", 1, 0.0075f, true);
	plot_eval(_FResDir + "eval/");
}

// Process a sigle image
int HFSSegment::processImage(Mat &seg, Mat &img3u, float c, int min_size)
{
	Mat idx_mat, lab3u, mag1u, tmp;
	int num_css;

	idx_mat = getSLICIdx(img3u, num_css);
	cv::cvtColor(img3u, lab3u, CV_BGR2Lab);
	mag_engine.process_img(img3u, mag1u);

	getSegmentationI(tmp, lab3u, mag1u, idx_mat, num_css, 0.08f, 100);
	getSegmentationII(seg, img3u, lab3u, mag1u, tmp, num_css, c, min_size);
	return num_css;
}

// Write matrix to binary file
bool HFSSegment::matWrite(CStr &filename, vecM &_M)
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
bool HFSSegment::matRead(CStr &filename, vecM &_M)
{
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	fread(buf, sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		std::cout << "Invalidate CvMat data file " << _S(filename) << endl;
		return false;
	}
	int matCnt;
	fread(&matCnt, sizeof(int), 1, f);
	_M.resize(matCnt);
	for (int i = 0; i < matCnt; i++) {
		int headData[3]; // Width, Height, Type
		fread(headData, sizeof(int), 3, f);
		Mat M(headData[1], headData[0], headData[2]);
		fread(M.data, sizeof(char), M.step * M.rows, f);
		M.copyTo(_M[i]);
	}
	std::fclose(f);
	return true;
}
