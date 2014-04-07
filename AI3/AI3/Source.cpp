

#include "Include.h"

using namespace cv;
using namespace std;

const int alphabetSize = 26;
const int ImageSize = 30;
const int trainingSet = 10;


char letters[alphabetSize] = { 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W' ,'X', 'Y', 'Z' };
int NumberEachRow[ImageSize];
Mat img, ImgRow;



// Number of training repetitions, the training data matrix and the training results matrix
const int reps = (alphabetSize*trainingSet);
Mat trainData(reps, ImageSize, CV_32F);
Mat trainResults(reps, alphabetSize, CV_32F);






int preprocess(){

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
				return 0;
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
	return 1;
}










void readPreprocessed(){

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
			if (!inputfile.is_open()){ cout << "cannot open file?" << endl; }
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

}
















int main( int argc, char** argv )	
{									
	// Det jeg(Alf) har skrevet her inn i main gjør om bildene til tallverdier som vi kan mate inn i ANN for å kjenne igjen bokstavene.
	// Jeg tror vi skal flytte dette til en egen funksjon som prepper, og dette er vel egentlig en engangsjobb for å gjøre klar tallene.

	if (preprocess() == 0){
		cout << "Something went wrong when preprocessing the files" << endl;
		return -1;
	}


    /*if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }*/
	
	readPreprocessed();


	
    return 0;
}
