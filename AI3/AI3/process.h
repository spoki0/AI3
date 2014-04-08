#pragma once
#include "Include.h"

int preprocess(const int ImageSize, const int alphabetSize, const int trainingSet, char letters[26]);
int readPreprocessed(cv::Mat& trainData, const int ImageSize, const int alphabetSize, const int trainingSet, char letters[26]);