#include "Include.h"
#include "process.h"

using namespace cv;
using namespace std;

// Static data related to input data
const int alphabetSize = 26;
char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
const int ImageSize = 256;

// Data for training
const int dataSet = 20;									// Total of data.
const int trainingSet = 20;								// Amount used for training.
const int trainingSamples = alphabetSize*trainingSet;	// Number for training.
const int testSamples = alphabetSize*dataSet;			// Number for testing.

// Related to the Neural net creation
const int attributes = ImageSize;	// Input til neural net
const int numberOfLayers = 3;		// Number of layers
const int sizeOfHiddenLayer = 615;	// Number of nodes on a given hidden layer, 615 
const int beta = 1;					//
const double alpha = 0.1;			// Sigmoid varaiables

int main( int argc, char** argv ) {

	//To avoid preprocessing the images all the time
	std::cout << "Have you preprocessed files already? Y/N ";
	char temp = toupper( getchar() ); std::cin.get();
	if (temp == 'N'){
		std::cout << "Preprocessing images";
		if (preprocess(ImageSize, alphabetSize, dataSet, letters) == 0){
			std::cout << "\nSomething went wrong when preprocessing the files" << endl;
			std::cin.get(); return -1;
		}
	}
	
	std::cout << "\nDo you need to make and train a new Neural Net? Y/N ";
	temp = toupper( getchar() ); 
	std::cin.get();

	if (temp == 'Y'){

		Mat training_set = Mat::zeros(trainingSamples, attributes,CV_32F);					//zeroed matrix to hold the training samples.
		Mat training_results = Mat::zeros(trainingSamples, alphabetSize, CV_32F);			//zeroed matrix to hold the training results.

		Mat test_set = Mat::zeros(testSamples,attributes,CV_32F);							//zeroed matrix to hold the test samples.
		Mat test_results = Mat::zeros(testSamples,alphabetSize,CV_32F);						//zeroed matrix to hold the test results.
	
		std::cout << "\nReading training data";
		if (readPreprocessed(training_set, training_results, ImageSize, alphabetSize, letters, (dataSet-dataSet+1), trainingSet) == 0) {
			std::cout << "\nSomething went wrong when opening preprocessed files" << endl;
			std::cin.get(); return -1;
		}
	
		std::cout << "\nReading test data";
		if (readPreprocessed(test_set, test_results, ImageSize, alphabetSize, letters, (dataSet-trainingSet+1), dataSet) == 0) {
			std::cout << "\nSomething went wrong when opening preprocessed files" << endl;
			std::cin.get(); return -1;
		}
	
		std::cout << "\nSetting up Neural Net";
	
		Mat layers(numberOfLayers, 1, CV_32S);

		layers.at<int>(0,0) = attributes;			//input layer
		layers.at<int>(1,0) = sizeOfHiddenLayer;	//hidden layer
		layers.at<int>(2,0) = alphabetSize;			//output layer
	
		//create the neural network.
		CvANN_MLP NeuralNet(layers, CvANN_MLP::SIGMOID_SYM,alpha,beta);

		// terminate the training after either 10 000
		// iterations or a very small change in the
		// network wieghts below the specified value

		// use backpropogation for training

		// co-efficents for backpropogation training
		// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
		CvANN_MLP_TrainParams params( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 10000, 0.005), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.5 );

		std::cout << "\nTraining Neural Net" << endl;
	
		int iterations = NeuralNet.train(training_set, training_results, Mat(), Mat(), params);
		std::cout << "\nCompleted after " << iterations << " iterations trough the training data set." << endl;

		// Save the model generated into an xml file.
		std::cout << "\nWriting to file param.xml ..." << endl;
		CvFileStorage* storage = cvOpenFileStorage( "param.xml", 0, CV_STORAGE_WRITE );
		NeuralNet.write(storage,"DigitOCR");
		cvReleaseFileStorage(&storage);
		std::cout << "\t\t ...Done." << endl; std::cin.get();

		cv::Mat classificationResult(1, alphabetSize, CV_32F);
		// Test the generated model with the test samples.
		cv::Mat test_sample;
		//count of correct prediction
		int correct_class = 0;
		//count of wrong prediction
		int wrong_class = 0;
 
		//classification matrix gives the count of classes to which the samples were classified.
		int classification_matrix[alphabetSize][alphabetSize] = {{}};
  
		// for each sample in the test set.
		for (int tsample = 0; tsample < testSamples; tsample++) {
 
			// extract the sample
 
			test_sample = test_set.row(tsample);
 
			//try to predict its class
 
			NeuralNet.predict(test_sample, classificationResult);
			/*The classification result matrix holds weightage  of each class.
			we take the class with the highest weightage as the resultant class */
 
			// find the class with maximum weightage.
			int maxIndex = 0;
			float value = 0.0f;
			float maxValue = classificationResult.at<float>(0,0);
			for(int index = 1; index < alphabetSize; index++) {   
				value = classificationResult.at<float>(0,index);
				if( value > maxValue )
				{   
					maxValue = value;
					maxIndex=index;
				}
			}
 
			//Now compare the predicted class to the actural class. if the prediction is correct then\
			//test_set_classifications[tsample][ maxIndex] should be 1.
			//if the classification is wrong, note that.
			if (test_results.at<float>(tsample, maxIndex)!=1.0f) {
				// if they differ more than floating point error => wrong class
				wrong_class++;
 
				//find the actual label 'class_index'
				for(int class_index = 0; class_index < alphabetSize; class_index++)
				{
					if(test_results.at<float>(tsample, class_index)==1.0f)
					{
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
		
		//Getting the % of correct and wrong characters
		cout << "Number of correct letters: " << correct_class << " / ";
		cout << correct_class*100.f/testSamples << "% \n";
		cout << "Number of wrong lettesr: " << wrong_class << " / "; 
		cout << wrong_class*100.f/testSamples << "%\n";
		cin.get();

		//Writing a 2d matrix that shows what the ANN guessed
		for (int i = 0; i < alphabetSize; i++) {
	        std::cout << "\t" << char(i+'A');
	    }
	    std::cout<<"\n\n";
		for(int row = 0; row < alphabetSize; row++) {
			std::cout << row << "\t";
			for(int col = 0; col < alphabetSize; col++) {
				std::cout << classification_matrix[row][col]<<"\t";
	        }
	        std::cout<<"\n\n";
	    }

	//No need to train
	} else {

		//read the model from the XML file and create the neural network.
		cout << "\nEnter the path to the stored xml file of the Neural Net: ";
		string path; getline( cin, path );

		CvANN_MLP NeuralNet;
		CvFileStorage* storage = cvOpenFileStorage( path.c_str(), 0, CV_STORAGE_READ );
		CvFileNode *n = cvGetFileNodeByName(storage,0,"DigitOCR");
		NeuralNet.read(storage,n);
		cvReleaseFileStorage(&storage);

		//reading a single preprocessedfile for predicting.
		cout << "\nEnter path to file for testing: ";
		getline( cin, path );
		Mat data = Mat::zeros(2, attributes,CV_32F);		//Zeroed matrix for single test
		Mat goal = Mat::zeros(1, alphabetSize, CV_32F);		//Zeroed matrix for result
		if (readPreprocessed( data, goal, ImageSize, path) == 0){
			std::cout << "\nSomething went wrong while reading file." << endl;
			cin.get(); return 1;
		}
		
		//prediction
		Mat prediction = Mat::zeros(1, alphabetSize, CV_32F); //Zeroed matrix for prediction
		NeuralNet.predict(data, prediction);

		//converting prediction to human readable.
		float maxres = 0;
		float maxtar = 0;
		int numres = 0;
		int numtar = 0;
		for(int z = 0; z < alphabetSize; z++) {
			if (maxres <= prediction.at<float>(0, z)) { 
				maxres = prediction.at<float>(0, z); 
				numres = z;
			};
			if (maxtar <= goal.at<float>(0, z)) { 
				maxtar = goal.at<float>(0, z); 
				numtar = z;
			}
		}
		std::cout << "\nPrediction: " << char(numres+'A') << " target is: " << char(numtar+'A');
	}

	std::cin.get();
    return 0;
}