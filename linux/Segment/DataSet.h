
#pragma once

#include "Stdafx.h"

class DataSet
{
	int loadStrList(CStr fileName, vecS &str) {
		ifstream fIn(fileName);
		string line;
		while (getline(fIn, line) && line.size())
			str.push_back(line);
		return str.size();
	}

public:
	CStr WkDir, ImgDir, ResDir;
	CStr gtDir, gtBdryDir, gtSegsDir;
	vecS testSet, trainSet, valSet;
	int testNum, trainNum, valNum;

	DataSet() {}

	DataSet(CStr fileName):
		WkDir(fileName),                       // the root directory of the data set
		ImgDir(fileName + "JPEGImages/%s.jpg"),// image folder
		ResDir(fileName + "Results/"),         // the folder to store the results
		gtDir(fileName + "Annotations/"),      // ground truth
		gtBdryDir(fileName + "Annotations/Boundaries/"),  // boudary ground truth
		gtSegsDir(fileName + "Annotations/Segmentation/") // region ground truth
	{
		printf("Process data in `%s'\n", fileName.c_str());
		testNum = loadStrList(fileName + "ImageSets/Main/test.txt", testSet);    // load file names of test images
		trainNum = loadStrList(fileName + "ImageSets/Main/train.txt", trainSet); // load file names of training images
		valNum = loadStrList(fileName + "ImageSets/Main/val.txt", valSet);       // load file names of validation images
	}
	
};