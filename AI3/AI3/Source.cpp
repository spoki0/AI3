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



#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
    /*if( argc != 2)
    {
     cout <<" Usage: display_image ImageToLoadAndDisplay" << endl;
     return -1;
    }*/

    Mat image, bwimg;
    image = imread("A1.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
	bwimg = imread("A1.jpg", CV_LOAD_IMAGE_GRAYSCALE);


    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
		cin.get();
        return -1;
    }

	int TotalNumberOfPixels = bwimg.rows * bwimg.cols;
	int ZeroPixels = TotalNumberOfPixels - countNonZero(bwimg);
	cout << "The number of pixels that are zero is " << ZeroPixels << endl;



    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.


    waitKey();                                          // Wait for a keystroke in the window
    return 0;
}


/*
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\core\core.hpp>
#include <cv.h>

using namespace cv;


int main()
{
    Mat rgbImage, grayImage, resizedImage, bwImage, result;

    rgbImage = imread("A1.jpg");
    cvtColor(rgbImage, grayImage, CV_RGB2GRAY);

    resize(grayImage, resizedImage, Size(grayImage.cols/3,grayImage.rows/4),
           0, 0, INTER_LINEAR);

    imwrite("A21.jpg", resizedImage);
    bwImage = imread("A21.jpg");
    threshold(bwImage, bwImage, 120, 255, CV_THRESH_BINARY);
    imwrite("A22.jpg", bwImage);
    imshow("Original", rgbImage);
    imshow("Resized", resizedImage);
    imshow("Resized Binary", bwImage);

    waitKey();
    return 0;
}


*/