#include "Include.h"

using namespace cv;
using namespace std;

const int alphabetSize = 26;
const int ImageSize = 30;
const int trainingSet = 20;

char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
int NumberEachRow[ImageSize];
Mat img, ImgRow;



// Number of training repetitions, the training data matrix and the training results matrix
const int reps = (alphabetSize*trainingSet);
Mat trainData(reps, ImageSize, CV_32F);
Mat trainResults(reps, alphabetSize, CV_32F);





int preprocess(){

	for (int i = 0; i < ImageSize; i++)  {
		NumberEachRow[i] = 0;
	}
	
	for (int i = 0; i < alphabetSize; i++){
		for (int samples = 1; samples <= trainingSet; samples++){
			
			string imgPath;
			stringstream PathPrep;
			PathPrep << "Resized_30x30/" << letters[i] << samples << ".jpg";
			imgPath = PathPrep.str();
			img = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);	// Read the file

			if (!img.data)										// Check for invalid input
			{
				cout << "Could not open or find the image" << endl;
				cin.get();
				return 0;
			}

			for (int x = 0; x < img.rows; x++)
			{
				ImgRow = img.row(x);	// Retrieves a row in the image
				NumberEachRow[x] = countNonZero(ImgRow);	// Counts the nonZero values in the row and stores the result in a array.
			}

			ofstream dataOut;
			string dataPath;
			stringstream dataPathPrep;
			dataPathPrep << "preprocessDataSet/" << letters[i] << samples;
			dataPath = dataPathPrep.str();
			dataOut.open(dataPath);

			for (int x = 0; x < img.rows; x++){
				dataOut << NumberEachRow[x] << " ";
			}

			dataOut << letters[i];
			dataOut.close();
		}
	}
	return 1;
}








int readPreprocessed(){

	int counter = 0;

	// All the files
	for(int i = 0; i < alphabetSize; i++){
		for(int j = 1; j <= trainingSet; j++){
			
			//get a proper filepath.
			string filepath;
			stringstream temp;
			temp << "preprocessDataSet/" << letters[i] << j;
			filepath = temp.str();

			ifstream inputfile(filepath);
			if (!inputfile.is_open()){ cout << "cannot open file?" << endl; return 0;}
			else {

				//Reading the data into the matrix.
				//One line represent one training event, and the values respond thusly.
				for (int x = 0; x < ImageSize; x++){
					inputfile >> trainData.at<int>(counter, x);
				}
			
				//This one is more fun. read the resulting character
				//convert it to an int value, and set the corresponding node
				//in the matrix to 1. example result is A, then pos 0 in the matrix is 1
				//if the result is G, pos 6 is 1.
				char number;
				inputfile >> number;
				trainData.at<int>(counter, int(number-'A') ) = 1;

				// close current file and increase row for both matrixes by 1.
				inputfile.close();
				counter++;
			}	
		}
	}
	return 1;
}


int main( int argc, char** argv ) {

	if (preprocess() == 0){
		cout << "Something went wrong when preprocessing the files" << endl;
		cin.get(); return -1;
	}
	
	if (readPreprocessed() == 0){
		cout << "Something went wrong when opening preprocessed files" << endl;
		cin.get(); return -1;
	}




	const int TRAINING_SAMPLES = 3050;
	const int ATTRIBUTES = 256;
	const int CLASSES = 26;
	const int TEST_SAMPLES = 1170;

	//matrix to hold the training sample
    Mat training_set(TRAINING_SAMPLES,ATTRIBUTES,CV_32F);
    //matrix to hold the labels of each taining sample
    Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);
    //matric to hold the test samples
    Mat test_set(TEST_SAMPLES,ATTRIBUTES,CV_32F);
    //matrix to hold the test labels.
    Mat test_set_classifications(TEST_SAMPLES,CLASSES,CV_32F);
 
    Mat classificationResult(1, CLASSES, CV_32F);

	cv::Mat layers(3,1,CV_32S);
    layers.at<int>(0,0) = ATTRIBUTES;//input layer
    layers.at<int>(1,0)=16;//hidden layer
    layers.at<int>(2,0) =CLASSES;//output layer

	 //create the neural network.
     //for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
     CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM,0.6,1);
 
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
