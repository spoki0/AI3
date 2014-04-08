#include "Include.h"
#include "process.h"

using namespace cv;
using namespace std;

const int alphabetSize = 26;
const int ImageSize = 30;
const int dataSet = 10;
const int trainingSet = 20;
const int trainingSamples = 260;
const int attributes = 30;
const int testSamples = 260;
const int sizeOfHiddenLayer = 16;
const int numberOfLayers = 3;
const int beta = 1;
const double alpha = 0.6;

char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
Mat img, ImgRow;

// Number of training repetitions, the training data matrix and the training results matrix
const int reps = (alphabetSize*trainingSet);
Mat trainData(reps, ImageSize, CV_32F);
Mat trainResults(reps, alphabetSize, CV_32F);

int main( int argc, char** argv ) {

	if (preprocess(ImageSize, alphabetSize, trainingSet, letters) == 0){
		cout << "Something went wrong when preprocessing the files" << endl;
		cin.get(); return -1;
	}
	
	if (readPreprocessed(trainData, ImageSize, alphabetSize, trainingSet, letters) == 0){
		cout << "Something went wrong when opening preprocessed files" << endl;
		cin.get(); return -1;
	}

    Mat training_set(trainingSamples,attributes,CV_32F);					//matrix to hold the training sample.
    Mat training_set_classifications(trainingSamples, alphabetSize, CV_32F);//matrix to hold the labels of each training sample.
    Mat test_set(testSamples,attributes,CV_32F);							//matrix to hold the test samples.
    Mat test_set_classifications(testSamples,alphabetSize,CV_32F);			//matrix to hold the test labels.
 
    Mat classificationResult(1, alphabetSize, CV_32F);

	Mat layers(numberOfLayers,1,CV_32S);
    layers.at<int>(0,0) = attributes;			//input layer
    layers.at<int>(1,0) = sizeOfHiddenLayer;	//hidden layer
    layers.at<int>(2,0) = alphabetSize;			//output layer

	//create the neural network.
    CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,alpha,beta);
 
    CvANN_MLP_TrainParams params(                                   
	// terminate the training after either 1000
	// iterations or a very small change in the
	// network wieghts below the specified value
	cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001),
	// use backpropogation for training
	CvANN_MLP_TrainParams::BACKPROP,
	// co-efficents for backpropogation training
	// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
	0.1,
	0.1
	);

    return 0;
}