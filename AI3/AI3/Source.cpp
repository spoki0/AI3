#include "Include.h"
#include "process.h"

using namespace cv;
using namespace std;

const int alphabetSize = 26;	// The size of the alfabet
const int ImageSize = 30;		// The size of the image (30x30)
const int dataSet = 10;			//  
const int trainingSet = 20;		// 20 because it is 20 of each letter in total
const int trainingSamples = 260;	// 
const int attributes = 30;			// 
const int testSamples = 260;		// Number of letters to be "learned" 10 of each letter
const int sizeOfHiddenLayer = 16;	// Nomber of hidden layers
const int numberOfLayers = 3;		// The number of layers
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

	printf( "\nUsing training dataset\n");
    int iterations = nnetwork.train(training_set, training_set_classifications,cv::Mat(),cv::Mat(),params);
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
    int classification_matrix[CLASSES][CLASSES]={{}};
 
    // for each sample in the test set.
    for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {
 
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
        for(int index=1;index<CLASSES;index++) {
			value = classificationResult.at<float>(0,index);

            if(value>maxValue) {
				maxValue = value;
                maxIndex=index;
			}
        }
 
        printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);
 
        //Now compare the predicted class to the actural class. if the prediction is correct then\
        //test_set_classifications[tsample][ maxIndex] should be 1.
        //if the classification is wrong, note that.
        if (test_set_classifications.at<float>(tsample, maxIndex)!=1.0f) {

            // if they differ more than floating point error => wrong class
 
            wrong_class++;
 
            //find the actual label 'class_index'
            for(int class_index=0;class_index<CLASSES;class_index++) {

                if(test_set_classifications.at<float>(tsample, class_index)==1.0f) {
 
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
    correct_class, (double) correct_class*100/TEST_SAMPLES,
    wrong_class, (double) wrong_class*100/TEST_SAMPLES);
    cout<<"   ";

    for (int i = 0; i < CLASSES; i++) {

        cout<< i<<"\t";
    }
    cout<<"\n";

    for(int row=0;row<CLASSES;row++) {
		
		cout<<row<<"  ";
        for(int col=0;col<CLASSES;col++) {

            cout<<classification_matrix[row][col]<<"\t";
        }
        cout<<"\n";
    }
 
    return 0;
}