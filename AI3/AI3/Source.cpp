#include "Include.h"
#include "process.h"

using namespace cv;
using namespace std;


// Static data related to input data
const int alphabetSize = 26;
char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
const int ImageSize = 256;

// Data for training
const int iterations = 10;
const int dataSet = 20;			// Total of data.
const int trainingSet = 10;		// Amount used for testing.
const int trainingSamples = alphabetSize*trainingSet;			// Number for training.
const int testSamples = (alphabetSize*dataSet)-trainingSamples;	// Number for testing.


// Related to the Neural net creation
const int attributes = ImageSize;	// Input til neural net
const int numberOfLayers = 3;		// Number of layers
const int sizeOfHiddenLayer = 64;	// Number of nodes on a given hidden layer
const int beta = 1;					// Sigmoid varaiables
const double alpha = 0.6;


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
	
    Mat training_set = Mat::zeros(trainingSamples, attributes,CV_32F);					//zeroed matrix to hold the training samples.
    Mat training_results = Mat::zeros(trainingSamples, alphabetSize, CV_32F);			//zeroed matrix to hold the training results.

    Mat test_set = Mat::zeros(testSamples,attributes,CV_32F);							//zeroed matrix to hold the test samples.
    Mat test_results = Mat::zeros(testSamples,alphabetSize,CV_32F);						//zeroed matrix to hold the test results.
	
	std::cout << "\nReading training data";
	if (readPreprocessed(training_set, training_results, ImageSize, alphabetSize, letters, (dataSet-dataSet+1), trainingSet) == 0){
		std::cout << "\nSomething went wrong when opening preprocessed files" << endl;
		std::cin.get(); return -1;
	}
	
	std::cout << "\nReading test data";
	if (readPreprocessed(test_set, test_results, ImageSize, alphabetSize, letters, (dataSet-trainingSet+1), dataSet) == 0){
		std::cout << "\nSomething went wrong when opening preprocessed files" << endl;
		std::cin.get(); return -1;
	}
	

	std::cout << "\nSetting up Neural Net";
	
	Mat layers(numberOfLayers,1,CV_32S);
    layers.at<int>(0,0) = attributes;			//input layer
	layers.at<int>(1,0) = sizeOfHiddenLayer;	//hidden layer
    layers.at<int>(2,0) = alphabetSize;			//output layer
	

/*	vector<unsigned> topology;
	std::cout << ".";
	topology.push_back(attributes);				//Input layer
	std::cout << ".";
	topology.push_back(sizeOfHiddenLayer);		//Hidden layer
	std::cout << ".";
	topology.push_back(alphabetSize);			//Output layer
	Net myNet(topology);

	
	//Iterate trough all the test samples.
	std::cout << "\nTraining the neural net";
	for(int z = 0; z < iterations; z++){
		for(int x = 0; x < testSamples; x++){

			//Convert input values
			vector<double> inputVals;
			for(int y = 0; y < attributes; y++){
				inputVals.push_back( test_set.at<int>(x, y) );
			}

			//Convert target values
			vector<double> targetVals;
			for(int y = 0; y < alphabetSize; y++){
				targetVals.push_back( test_results.at<int> (x, y) );
			}

			myNet.feedForward(inputVals);				//Input values to neural net
			myNet.backProp(targetVals);					//Specify target output and update

		}
		std::cout << ".";
	}

	std::cout << "\nTesting neural net";
	for(int x = 0; x < trainingSamples; x++){

		//Convert input values
		vector<double> inputVals;
		for(int y = 0; y < attributes; y++){
			inputVals.push_back( training_set.at<int>(x, y) );
		}

		//Convert target values
		vector<double> targetVals;
		for(int y = 0; y < alphabetSize; y++){
			targetVals.push_back( training_results.at<int> (x, y) );
		}


		myNet.feedForward(inputVals);				//Input values to neural net

		vector<double> results;
		myNet.getResults(results);					//Specify target output and update

		float maxres = 0;
		float maxtar = 0;
		int numres = 0;
		int numtar = 0;

		for(int z = 0; z < results.size(); z++){
			if (maxres <= results[z]){ 
				maxres = results[z]; 
				numres = z; };
			if (maxtar <= targetVals[z]){ 
				maxtar = targetVals[z]; 
				numtar = z; }
		
		}
		std::cout << "\nPrediction: " << numres << char(numres+'A') << " target is: " << numtar << char(numtar+'A');
	}
	cout << endl << "1st run med ny kode" << endl;
	cin.get();

	*/


	
	//create the neural network.
    CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,alpha,beta);

	// terminate the training after either 10 000
	// iterations or a very small change in the
	// network wieghts below the specified value

	// use backpropogation for training

	// co-efficents for backpropogation training
	// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
    CvANN_MLP_TrainParams params( cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 0.000001), CvANN_MLP_TrainParams::BACKPROP, 0.1, 0.1 );

	std::cout << "\nTraining Neural Net" << endl;
	
    int iterations = nnetwork.train(training_set, training_results, Mat(), Mat(), params);
	std::cout << "\nCompleted after " << iterations << " iterations trough the training data set." << endl;

 
    // Save the model generated into an xml file.
	std::cout << "Writing to file..." << endl;
    CvFileStorage* storage = cvOpenFileStorage( "param.xml", 0, CV_STORAGE_WRITE );
    nnetwork.write(storage,"DigitOCR");
    cvReleaseFileStorage(&storage);
	std::cout << "\t\t ...Done." << endl; std::cin.get();

	/*
	std::cout << "Testing the Neural Net against data set." << endl;
	//Run all the tests :D
	int correct = 0;
	int wrong   = 0;
	for(int x = 0; x < testSamples; x++){

		//Checked, they hold valid data.
		Mat test_row = test_set.row(x);							// get test row number x
		Mat result_row = Mat::zeros(1, alphabetSize, CV_32F);	// results row; all zeroed out.

		//Commence testing.
		nnetwork.predict(test_row, result_row);

		
		for(int y = 0; y < attributes; y++){
			std::cout << test_row.at<int>(0, y) << " ";
		} cout << endl;
		
		for(int y = 0; y < alphabetSize; y++){
			std::cout << result_row.at<float>(0, y) << " ";
		} cout << endl;
		
		for(int y = 0; y < alphabetSize; y++){
			std::cout << test_results.at<int>(x, y) << " ";
		} cout << endl;
		cin.get();
		
		

		float maxres = 0;
		float maxtar = 0;
		int numres = 0;
		int numtar = 0;
		for (int y = 0; y < alphabetSize; y++){

			//printing the values for this row. All extremely close to zero and identical.
			//std::cout << result_row.at<float>(0, y) << " ";

			//Find highest value of results.
			if (maxres < result_row.at<float>(0, y)){
				maxres = result_row.at<float>(0, y);
				numres = y;
			}
			//Find highest value of target.
			if (maxtar < test_results.at<float>(x, y)){
				maxtar = test_results.at<float>(x, y);
				numtar = y;
			}
		}
		
		std::cout << "prediction: " << char(numres+'A') << " target is: " << char(numtar+'A') << endl;
		//cin.get();

		if (numres == numtar){ correct++; }
		else { wrong++; }

	}
	std::cout << "\nOut of " << testSamples << " the Neural net got " << correct << " correct and " << wrong << " wrong." << endl;
	
	


	// Found it to be a tad complicated, so I wrote my own part above. :D
	// Don't even know what the bottom part does..

	// Test the generated model with the test samples.
 //   cv::Mat test_sample;
 //   int correct_class = 0;	//count of correct classifications
 //   int wrong_class = 0;	//count of wrong classifications
 //
 //   //classification matrix gives the count of classes to which the samples were classified.
 //   int classification_matrix[alphabetSize][alphabetSize]={{}};
 //
 //   // for each sample in the test set.
 //   for (int tsample = 0; tsample < testSamples; tsample++) {
 //
 //       // extract the sample
 //       test_sample = test_set.row(tsample);
 //
 //       //try to predict its class
 //       nnetwork.predict(test_sample, classificationResult);
 //       /*The classification result matrix holds weightage  of each class.
 //       we take the class with the highest weightage as the resultant class */
 //
 //       // find the class with maximum weightage.
 //       int maxIndex = 0;
 //       float value = 0.0f;
 //       float maxValue = 0;
 //       for(int index=0; index < alphabetSize; index++) {
	//		value = classificationResult.at<float>(0,index);

 //           if(value>=maxValue) {
	//			maxValue = value;
 //               maxIndex=index;
	//		}
 //       }

 //
 //       std::cout << "Testing Sample " << tsample << " -> class result " << maxIndex << endl;
	//	cin.get();
 //
 //       //Now compare the predicted class to the actural class. if the prediction is correct then\
 //       //test_results[tsample][ maxIndex] should be 1.
 //       //if the classification is wrong, note that.
 //       if (test_results.at<float>(tsample, maxIndex)!=1.0f) {

 //           // if they differ more than floating point error => wrong class
 //
 //           wrong_class++;
 //
 //           //find the actual label 'class_index'
 //           for(int class_index=0;class_index<alphabetSize;class_index++) {

 //               if(test_results.at<float>(tsample, class_index)==1.0f) {
 //
 //                   classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
 //                   break;
 //               }
 //           }
 //
 //       } else {
 //
 //           // otherwise correct
 //           correct_class++;
 //           classification_matrix[maxIndex][maxIndex]++;
 //       }
 //   }
 //
 //   printf( "\nResults on the testing dataset\n"
 //   "\tCorrect classification: %d (%g%%)\n"
 //   "\tWrong classifications: %d (%g%%)\n", 
 //   correct_class, (double) correct_class*100/testSamples,
 //   wrong_class, (double) wrong_class*100/testSamples);
 //   std::cout<<"   "; cin.get();

 //   for (int i = 0; i < alphabetSize; i++) {

 //       std::cout<< i<<"\t";
 //   }
 //   std::cout<<"\n"; cin.get();

 //   for(int row=0;row<alphabetSize;row++) {
	//	
	//	std::cout<<row<<"  ";
 //       for(int col=0;col<alphabetSize;col++) {

 //           std::cout<<classification_matrix[row][col]<<"\t";
 //       }
 //       std::cout<<"\n"; cin.get();
 //   }
//	cout << endl << "2nd run" << endl;
	
		cv::Mat classificationResult(1, alphabetSize, CV_32F);
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
			for(int index=1;index<alphabetSize;index++)
            {   value = classificationResult.at<float>(0,index);
                if(value>maxValue)
                {   maxValue = value;
                    maxIndex=index;
 
                }
            }
 
            //Now compare the predicted class to the actural class. if the prediction is correct then\
            //test_set_classifications[tsample][ maxIndex] should be 1.
            //if the classification is wrong, note that.
            if (test_results.at<float>(tsample, maxIndex)!=1.0f)
            {
                // if they differ more than floating point error => wrong class
 
                wrong_class++;
 
                //find the actual label 'class_index'
				for(int class_index=0;class_index<alphabetSize;class_index++)
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
 
        printf( "\nResults on the testing dataset\n"
        "\tCorrect classification: %d (%g%%)\n"
        "\tWrong classifications: %d (%g%%)\n", 
        correct_class, (double) correct_class*100/testSamples,
        wrong_class, (double) wrong_class*100/testSamples);
        
	std::cin.get();
    return 0;
}














