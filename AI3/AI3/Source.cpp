// Neural net :D
/*
#include "Include.h"
// #include "Net.h"

//using namespace std;

int main(){

	
	vector<unsigned> topology;
	topology.push_back(3);
	topology.push_back(2);
	topology.push_back(1);
	Net myNet(topology);

	vector<double> inputVals;
	myNet.feedForward(inputVals);

	vector<double> targetVals;
	myNet.backProp(targetVals);

	vector<double> resultVals;
	myNet.getResults(resultVals);
	
}
*/


#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const int alphabetSize = 26;
const int ImageSize = 30;
const int trainingSet = 10;


int main( int argc, char** argv )	
{									
	// Det jeg(Alf) har skrevet her inn i main gjør om bildene til tallverdier som vi kan mate inn i ANN for å kjenne igjen bokstavene.
	// Jeg tror vi skal flytte dette til en egen funksjon som prepper, og dette er vel egentlig en engangsjobb for å gjøre klar tallene.

	char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
	int NumberEachRow[ImageSize];
	Mat img, ImgRow;

    /*if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }*/
	
	for (int i = 0; i < alphabetSize; i++){
		for (int samples = 1; samples <= trainingSet; samples++){
			string imgPath;
			stringstream PathPrep;
			PathPrep << "Resized_30x30/" << letters[i] << samples << ".jpg";
			imgPath = PathPrep.str();
			img = imread(imgPath, CV_LOAD_IMAGE_GRAYSCALE);	// Read the file

			if (!img.data)										// Check for invalid input
			{
				cout << "Could not open or find the image" << std::endl;
				cin.get();
				return -1;
			}

			for (int x = 0; x < img.rows; x++)
			{
				ImgRow = img.row(i);	// Retrieves a row in the image
				NumberEachRow[i] = countNonZero(ImgRow);	// Counts the nonZero values in the row and stores the result in a array.

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
	
    return 0;
}
