#include "process.h"
using namespace std;
using namespace cv;





int preprocess(const int ImageSize, const int alphabetSize, const int trainingSet, char letters[26]){

	Mat img, ImgRow;
	int NumberEachRow[30];

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

			//Sharpen the shait
			cv::Mat tmp;
			cv::GaussianBlur(img, tmp, cv::Size(5,5), 5);
			cv::addWeighted(img, 1.5, tmp, -0.5, 0, img);



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









int readPreprocessed(cv::Mat &trainData, cv::Mat &trainResults, const int ImageSize, const int alphabetSize, char letters[26], const int start, const int stop){

	int counter = 0;

	// All the files
	for(int j = start; j <= stop; j++){
		for(int i = 0; i < alphabetSize; i++){
			
			//get a proper filepath.
			string filepath;
			stringstream temp;
			temp << "preprocessDataSet/" << letters[i] << j;
			filepath = temp.str();
	
			//if you can open file.
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
				trainResults.at<int>(counter, int(number-'A') ) = 1;

				// close current file and increase row for both matrixes by 1.
				inputfile.close();
				counter++;
			}	
		}
		std::cout << ".";
	}
	std::cout << endl;
	return 1;
}