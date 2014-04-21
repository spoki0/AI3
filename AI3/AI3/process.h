#pragma once
#include "Include.h"

//Preprocess jpg files into binary data that can be quickly read later
int preprocess(const int ImageSize, const int alphabetSize, const int trainingSet, char letters[26]);

//Read the set of preprocessed data.
int readPreprocessed(cv::Mat &trainData, cv::Mat &trainResults, const int ImageSize, const int alphabetSize, char letters[26], const int start, const int stop);

//Read a single file of an preprosessed image.
int readPreprocessed(cv::Mat &trainData, cv::Mat &trainResults, const int ImageSize, std::string filepath);