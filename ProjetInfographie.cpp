/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
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


// Apply a Gaussian blur to the entire training sample with 4 levels
static void blurImages(Mat img, int label, vector<Mat>& images, vector<int>& labels)
{
	for (int i = 11; i < 51; i+= 10)
	{
		Mat img2 = img.clone();

		GaussianBlur(img, img2, Size(i, i), 0);
		images.push_back(img2);
		labels.push_back(label);
	}
}

static Mat norm_0_255(InputArray _src) {
	Mat src = _src.getMat();
	// Create and return normalized image:
	Mat dst;
	switch (src.channels()) {
	case 1:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		break;
	case 3:
		cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			Mat img = imread(path, 0);
			int label = atoi(classlabel.c_str());
			images.push_back(img);
			labels.push_back(label);

			blurImages(img, label, images, labels);
		}
	}
}
int main(int argc, const char* argv[]) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.
	if (argc < 2) {
		cout << "usage: " << argv[0] << " <csv.ext> <output_folder> " << endl;
		exit(1);
	}
	string output_folder = ".";
	if (argc == 3) {
		output_folder = string(argv[2]);
	}
	// Get the path to your CSV.
	string fn_csv = string(argv[1]);
	// These vectors hold the images and corresponding labels.
	vector<Mat> images;
	vector<int> labels;
	// Read in the data. This can fail if no valid
	// input filename is given.
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (const cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Quit if there are not enough images for this demo.
	if (images.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(Error::StsError, error_message);
	}

	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size:
	int height = images[0].rows;



	Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();

	model->train(images, labels);

	// The following line predicts the label of a given
	// test image:
	Mat testSample;
	int testLabel;
	try
	{
		testSample = imread("imagesTest/4.jpg", 0);
		GaussianBlur(testSample, testSample, Size(51, 51), 0, 0);
		testLabel = 4;
	}
	catch (Exception e)
	{
		cout << "testSample not found" << endl;
	}

	int predictedLabel = model->predict(testSample);


	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	cout << result_message << endl;
	// Here is how to get the eigenvalues of this Eigenfaces model:
	Mat eigenvalues = model->getEigenValues();
	// And we can do the same to display the Eigenvectors (read Eigenfaces):
	Mat W = model->getEigenVectors();
	// Get the sample mean from the training data
	Mat mean = model->getMean();
	// Display or save:
	if (argc == 2) {
		imshow("mean", norm_0_255(mean.reshape(1, images[0].rows)));
	}
	else {
		imwrite(format("%s/mean.png", output_folder.c_str()), norm_0_255(mean.reshape(1, images[0].rows)));
	}
	// Display or save the first, at most 16 Fisherfaces:
	for (int i = 0; i < min(16, W.cols); i++) {
		string msg = format("Eigenvalue #%d = %.5f", i, eigenvalues.at<double>(i));
		cout << msg << endl;
		// get eigenvector #i
		Mat ev = W.col(i).clone();
		// Reshape to original size & normalize to [0...255] for imshow.
		Mat grayscale = norm_0_255(ev.reshape(1, height));
		// Show the image & apply a Bone colormap for better sensing.
		Mat cgrayscale;
		applyColorMap(grayscale, cgrayscale, COLORMAP_BONE);
		// Display or save:
		if (argc == 2) {
			imshow(format("fisherface_%d", i), cgrayscale);
		}
		else {
			imwrite(format("%s/fisherface_%d.png", output_folder.c_str(), i), norm_0_255(cgrayscale));
		}
	}
	// Display or save the image reconstruction at some predefined steps:
	for (int num_component = 0; num_component < min(16, W.cols); num_component++) {
		// Slice the Fisherface from the model:
		Mat ev = W.col(num_component);
		Mat projection = LDA::subspaceProject(ev, mean, images[0].reshape(1, 1));
		Mat reconstruction = LDA::subspaceReconstruct(ev, mean, projection);
		// Normalize the result:
		reconstruction = norm_0_255(reconstruction.reshape(1, images[0].rows));
		// Display or save:
		if (argc == 2) {
			imshow(format("fisherface_reconstruction_%d", num_component), reconstruction);
		}
		else {
			imwrite(format("%s/fisherface_reconstruction_%d.png", output_folder.c_str(), num_component), reconstruction);
		}
	}
	// Display if we are not writing to an output folder:
	if (argc == 2) {
		waitKey(0);
	}
	return 0;
}
