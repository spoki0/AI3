#include <stdio.h>	// For printf()
#include <cv.h>		// Main OpenCV library.
#include <highgui.h>	// OpenCV functions for files and graphical windows.

int main(int argc, char* argv[])
{
	// Open the file "lena.jpg".
	IplImage* img = cvLoadImage("A1.jpg", CV_LOAD_IMAGE_UNCHANGED);
	if (!img) {
		printf("Error: Could not open the image file! \n");
		exit(1);
	}

	cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
	cvShowImage("Original", img);

	// Blur the image.
	cvSmooth(img, img, CV_BLUR);

	// Save the blurred image to a file.
	cvSaveImage("blurred.jpg", img);

	// Show the blurred image on the screen.
	cvNamedWindow("Blurred", CV_WINDOW_AUTOSIZE);
	cvShowImage("Blurred", img);

	// Wait for the user to press something on the graphical window.
	// Note: cvWaitKey() is needed for time to draw on the screen.
	cvWaitKey(0);

	// Free the resources.
	//cvDestroyWindow("Blurred");
	cvDestroyAllWindows();
	cvReleaseImage( &img );

	return 0;
}