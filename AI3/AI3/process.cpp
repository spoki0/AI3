#include "process.h"
using namespace std;
using namespace cv;

int preprocess(const int ImageSize, const int alphabetSize, const int dataSet, char letters[26]){

	Mat img, imgResize, ImgRow;
	int pixelArray[256];

	for (int i = 0; i < alphabetSize; i++){
		for (int samples = 1; samples <= dataSet; samples++){
			
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

			//Sharpen image
			Mat tmp;
			GaussianBlur(img, tmp, cv::Size(5,5), 5);
			addWeighted(img, 1.5, tmp, -0.5, 0, img);


			//Resize to 16x16 and read in each pixel as white or not
			resize(img, imgResize, Size(16 ,16),0,0, CV_INTER_AREA);
			int a = 0;
			for(int x=0;x<16;x++){  
				for(int y=0;y<16;y++){
					
					pixelArray[a]=(img.at<uchar>(x,y)==255)?1:0;
					a++;
				}
			}
 
			ofstream dataOut;
			string dataPath;
			stringstream dataPathPrep;
			dataPathPrep << "preprocessDataSet/" << letters[i] << samples;
			dataPath = dataPathPrep.str();
			dataOut.open(dataPath);

			for (int x = 0; x < ImageSize; x++){
				dataOut << pixelArray[x] << " ";
			}

			dataOut << letters[i];
			dataOut.close();
		}
		std::cout << ".";
	}
	std::cout << endl;
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
				for (int x = 0; x < ImageSize; x++){
					inputfile >> trainData.at<float>(counter, x);
				}
				
				//This one is more fun. read the resulting character
				//convert it to a float value, and set the corresponding node
				//in the matrix to 1. example result is A, then pos 0 in the matrix is 1
				//if the result is G, pos 6 is 1.
				char number;
				inputfile >> number;
				trainResults.at<float>(counter, float(number-'A')) = 1.0;
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