#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp""

#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;

// Gaussian blur
// Blur input image with 4 kernel sizes
static void blurImages(Mat img, int label, vector<Mat>& images, vector<int>& labels)
{
	for (int i = 11; i < 51; i += 10)
	{
		Mat img2 = img.clone();
		GaussianBlur(img, img2, Size(i, i), 0, 0);
		images.push_back(img2);
		labels.push_back(label);
	}
}

// Open file then read all images and their label and add them to the images and labels vectors
// We also add 4 levels of blur for each image
// Copyright: adapted from OpenCV Face Recognition example (https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		cout << "No valid input file was given, please check the given filename." << endl;
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		// Read image path and label in file
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);

		if (!path.empty() && !classlabel.empty()) {
			Mat img = imread(path, 0);
			int label = atoi(classlabel.c_str());
			// Add image
			images.push_back(img);
			// Add label
			labels.push_back(label);
			// Add 4 blurred images of the original image
			blurImages(img, label, images, labels);
		}
	}
}

// Copyright: based on OpenCV Face Recognition example (https://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html)
int main(int argc, const char* argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> " << endl;
		exit(1);
	}

	// Get the path to your CSV.
	string fn_csv = string(argv[1]);

	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;

	// Read in the data. This can fail if no valid input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (const cv::Exception& e) {
		cout << "Error while reading file" << endl;
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		cout << "Needs at least 2 images to work. Please add more images to your data set!" << endl;
	}

	// Create a Face Recognizer then train it with input faces and all their levels of blur
	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
	cout << "Training face recognizer model, it might take few minutes ..." << endl;
	model->train(images, labels);
	cout << "Training done" << endl;

	Mat testSample;
	int testLabel;
	// Open the image to be labelled
	// The input image is not blurred at first
	// We blur the image with an arbitrary Gaussian blur kernel size then predict it's label
	string imgName;
	try
	{
		cout << "Choose the image to label (int between 0 & 5):" << endl;
		int num;
		cin >> num;

		imgName = "imagesTest/" + std::to_string(num) + ".jpg";
		testSample = imread(imgName, 0);
		GaussianBlur(testSample, testSample, Size(41, 41), 0, 0);
		imshow("Blurred test sample", testSample);
		testLabel = num;
	}
	catch (Exception e)
	{
		cout << "Image not found" << endl;
		exit(1);
	}

	// The following line predicts the label of a given test image:
	int predictedLabel = model->predict(testSample);

	// predictedLabel is the label found by the FaceRecognizer
	// testLabel is the input label
	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;

	// Show original input image to compare with prediction
	imshow("Original input image before blur",imread(imgName, 0));

	// Show the first image in "faces" (training) directory of predicted label
	imgName = "faces/" + std::to_string(predictedLabel) + "/" + std::to_string(predictedLabel) + ".1.jpg";
	imshow("Image of predicted label",imread(imgName, 0));


	waitKey(0);

	return 0;
}