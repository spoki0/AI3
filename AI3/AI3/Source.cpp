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


// Related to the Neural net creation
const int attributes = ImageSize;	// Input til neural net
const int numberOfLayers = 3;		// Number of layers
const int sizeOfHiddenLayer = 16;	// Number of nodes on a given hidden layer
const int beta = 1;					// Sigmoid varaiables
const double alpha = 0.6;






int main( int argc, char** argv ) {

	if (preprocess(ImageSize, alphabetSize, trainingSet, letters) == 0){
		cout << "Something went wrong when preprocessing the files" << endl;
		cin.get(); return -1;
	}
	

    Mat training_set(trainingSamples, attributes,CV_32F);					//matrix to hold the training sample.
    Mat training_results(trainingSamples, alphabetSize, CV_32F);			//matrix to hold the labels of each training sample.

    Mat test_set(testSamples,attributes,CV_32F);							//matrix to hold the test samples.
    Mat test_results(testSamples,alphabetSize,CV_32F);						//matrix to hold the test labels.
 
	if (readPreprocessed(training_set, training_results, ImageSize, alphabetSize, trainingSet, letters) == 0){
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






	printf( "\nUsing training dataset\n");
    int iterations = nnetwork.train(training_set, training_results,cv::Mat(),cv::Mat(),params);
    printf( "Training iterations: %i\n\n", iterations);
 
    // Save the model generated into an xml file.
    CvFileStorage* storage = cvOpenFileStorage( "param.xml", 0, CV_STORAGE_WRITE );
    nnetwork.write(storage,"DigitOCR");
    cvReleaseFileStorage(&storage);
 
    // Test the generated model with the test samples.
    cv::Mat test_sample;
    //count of correct classifications
    int correct_class = 0;
    //count of wrong classifications
    int wrong_class = 0;
 
    //classification matrix gives the count of classes to which the samples were classified.
    int classification_matrix[alphabetSize][alphabetSize]={{}};
 
    // for each sample in the test set.
    for (int tsample = 0; tsample < testSamples; tsample++) {
 
        // extract the sample
 
        test_sample = test_set.row(tsample);
 
        //try to predict its class
 
        nnetwork.predict(test_sample, classificationResult);
        /*The classification result matrix holds weightage  of each class.
        we take the class with the highest weightage as the resultant class */
 
        // find the class with maximum weightage.
        int maxIndex = 0;
        float value=0.0f;
        float maxValue=classificationResult.at<float>(0,0);
        for(int index=1;index<alphabetSize;index++) {
			value = classificationResult.at<float>(0,index);

            if(value>maxValue) {
				maxValue = value;
                maxIndex=index;
			}
        }
 
        printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);
 
        //Now compare the predicted class to the actural class. if the prediction is correct then\
        //test_results[tsample][ maxIndex] should be 1.
        //if the classification is wrong, note that.
        if (test_results.at<float>(tsample, maxIndex)!=1.0f) {

            // if they differ more than floating point error => wrong class
 
            wrong_class++;
 
            //find the actual label 'class_index'
            for(int class_index=0;class_index<alphabetSize;class_index++) {

                if(test_results.at<float>(tsample, class_index)==1.0f) {
 
                    classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
                    break;
                }
            }
 
        } else {
 
            // otherwise correct
 
            correct_class++;
            classification_matrix[maxIndex][maxIndex]++;
        }
    }
 
    printf( "\nResults on the testing dataset\n"
    "\tCorrect classification: %d (%g%%)\n"
    "\tWrong classifications: %d (%g%%)\n", 
    correct_class, (double) correct_class*100/testSamples,
    wrong_class, (double) wrong_class*100/testSamples);
    cout<<"   ";

    for (int i = 0; i < alphabetSize; i++) {

        cout<< i<<"\t";
    }
    cout<<"\n";

    for(int row=0;row<alphabetSize;row++) {
		
		cout<<row<<"  ";
        for(int col=0;col<alphabetSize;col++) {

            cout<<classification_matrix[row][col]<<"\t";
        }
        cout<<"\n";
    }
 
    return 0;
}