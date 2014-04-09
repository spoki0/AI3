#include "Include.h"
#include "process.h"

using namespace cv;
using namespace std;

// Static data related to input data
const int alphabetSize = 26;
char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
const int ImageSize = 30;

// Data for training
const int dataSet = 20;			// Total of data.
const int trainingSet = 10;		// Amount used for testing.
const int trainingSamples = alphabetSize*trainingSet;			// Number for training.
const int testSamples = (alphabetSize*dataSet)-trainingSamples;	// Number for testing.

const int attributes = ImageSize;	// Input til neural net


// Related to the Neural net creation
const int sizeOfHiddenLayer = 16;
const int numberOfLayers = 3;
const int beta = 1;				// Sigmoid varaiables
const double alpha = 0.6;


//Mat img, ImgRow;

// Number of training repetitions, the training data matrix and the training results matrix
const int reps = (alphabetSize*trainingSet);
Mat trainData(reps, ImageSize, CV_32F);
Mat trainResults(reps, alphabetSize, CV_32F);




int main( int argc, char** argv ) {

	if (preprocess(ImageSize, alphabetSize, trainingSet, letters) == 0){
		cout << "Something went wrong when preprocessing the files" << endl;
		cin.get(); return -1;
	}
	

    Mat training_set(trainingSamples, attributes,CV_32F);					//matrix to hold the training sample.
    Mat training_results(trainingSamples, alphabetSize, CV_32F);			//matrix to hold the labels of each training sample.

    Mat test_set(testSamples,attributes,CV_32F);							//matrix to hold the test samples.
    Mat test_results(testSamples,alphabetSize,CV_32F);						//matrix to hold the test labels.
 
	if (readPreprocessed(trainData, trainResults, ImageSize, alphabetSize, trainingSet, letters) == 0){
		cout << "Something went wrong when opening preprocessed files" << endl;
		cin.get(); return -1;
	}
	

    Mat classificationResult(1, alphabetSize, CV_32F);

	Mat layers(numberOfLayers,1,CV_32S);
    layers.at<int>(0,0) = attributes;			//input layer
    layers.at<int>(1,0) = sizeOfHiddenLayer;	//hidden layer
    layers.at<int>(2,0) = alphabetSize;			//output layer

	//create the neural network.
    CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,alpha,beta);
 


	// terminate the training after either 1000
	// iterations or a very small change in the
	// network wieghts below the specified value

	// use backpropogation for training

	// co-efficents for backpropogation training
	// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
    CvANN_MLP_TrainParams params( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1 );






    return 0;
}