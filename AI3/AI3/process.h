#pragma once
#include "Include.h"


int preprocess(const int ImageSize, const int alphabetSize, const int trainingSet, char letters[26]);
int readPreprocessed(cv::Mat &trainData, cv::Mat &trainResults, const int ImageSize, const int alphabetSize, char letters[26], const int start, const int stop);