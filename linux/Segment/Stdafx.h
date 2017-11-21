#pragma once

#pragma warning(disable: 4996)
#pragma warning(disable: 4995)
#pragma warning(disable: 4805)
#pragma warning(disable: 4819)
#pragma warning(disable: 4267)

#include <cstdlib>
#include <stdio.h>
#include <assert.h>
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iostream>
#define _USE_MATH_DEFINES
#include <cmath>
#include <time.h>
#include <fstream>
#include <limits>

#ifdef _WIN32
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#endif

#ifndef __APPLE__
#include <omp.h>
#endif

#ifndef _WIN32
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#endif

using namespace std;

#include <opencv2/opencv.hpp> 
#include <opencv/cv.h>

#define CV_VERSION_ID CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#ifdef _DEBUG
#define cvLIB(name) "opencv_" name CV_VERSION_ID "d"
#else
#define cvLIB(name) "opencv_" name CV_VERSION_ID
#endif

#pragma comment( lib, cvLIB("ts"))
#pragma comment( lib, cvLIB("world"))
using namespace cv;

typedef const Mat CMat;
typedef const string CStr;
typedef vector<Mat> vecM;
typedef vector<string> vecS;
typedef vector<int> vecI;
typedef vector<bool> vecB;
typedef vector<float> vecF;
typedef vector<double> vecD;

#ifndef _WIN32
typedef uint64_t UINT64;
typedef uint32_t UINT;
typedef int64_t INT64;
typedef uint8_t byte;
typedef bool BOOL;
#endif

#define _S(str) ((str).c_str())
#define CV_Assert_(expr, args) \
{\
    if (!(expr)) { \
	   string msg(args); \
	   printf("%s in %s:%d\n", msg.c_str(), __FILE__, __LINE__); \
	   cv::error(cv::Exception(CV_StsAssert, msg, __FUNCTION__, __FILE__, __LINE__)); \
	}\
}
#define CHK_IND(p) ((p).x >= 0 && (p).x < _w && (p).y >= 0 && (p).y < _h)

const Point DIRECTION4[5] = {
	Point(-1, 0), //Direction 0: left
	Point(0, -1), //Direction 1: up
	Point(1, 0),  //Direction 2: right
	Point(0, 1),  //Direction 3: bottom
	Point(0, 0),
};  //format: {dx, dy}

#include "../LibLinear/linear.h"
#include "gSLIC/gSLIC.h"
#include "CmFile.h"
#include "NVTimer.h"


#define DOUBLE_EPS 1E-6
#define SQRT_3     1.732050807568877

const Point CIRCLE3[29] = {
	Point(0, 1), Point(0, 2), Point(0, 3), Point(1, 1), Point(1, 2), Point(2, 1), Point(2, 2),
	Point(1, 0), Point(2, 0), Point(3, 0), Point(1, -1), Point(1, -2), Point(2, -1), Point(2, -2),
	Point(0, -1), Point(0, -2), Point(0, -3), Point(-1, -1), Point(-1, -2), Point(-2, -1), Point(-2, -2),
	Point(-1, 0), Point(-2, 0), Point(-3, 0), Point(-1, 1), Point(-1, 2), Point(-2, 1), Point(-2, 2),
	Point(0, 0)
};
const Point CIRCLE2[13] = {
	Point(0, 1), Point(0, 2), Point(1, 1),
	Point(1, 0), Point(2, 0), Point(1, -1),
	Point(0, -1), Point(0, -2), Point(-1, -1),
	Point(-1, 0), Point(-2, 0), Point(-1, 1),
	Point(0, 0)
};

