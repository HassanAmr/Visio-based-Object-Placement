//#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
//#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <string>
#include <stdio.h>
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "Image/Image.h"
#include "Image/ImageIO.h"
#include "Fitline/LFLineFitter.h"
#include "Fdcm/LMLineMatcher.h"

//using namespace cv;
double* lineRep;

cv::String SplitFilename (const std::string& str)
{
  std::size_t found = str.find_last_of("/\\");
  return str.substr(found+1);
}

cv::String RemoveFileExtension (const std::string& str)
{
  std::size_t lastindex = str.find_last_of(".");
  return str.substr(0, lastindex);
}


cv::Rect FdcmDetection(cv::Mat queryImg, int numberOfLineSegments)
{
	// Create Iamge
	Image<uchar> inputImage;
	inputImage.Resize(queryImg.cols,queryImg.rows,false);
	int row,col;
	for (col=0; col < queryImg.cols; col++)
	{
		for (row=0; row < queryImg.rows; row++)
		{
			inputImage.Access(col,row) = queryImg.at<uchar>(cv::Point(col,row));

		}
	}

	//TODO: allow to choose these parameters in a seperate window maybe

	double sigma_fit_a_line = 0.5;
	double sigma_find_support = 0.5;
	double max_gap = 2.0;
	double nLinesToFitInStage1 = 0;
	double nTrialsPerLineInStage1 = 0;
	double nLinesToFitInStage2 = 100000;
	double nTrialsPerLineInStage2 = 1;


	int nLayer = 2;
	int nLinesToFitInStage[2];
	int nTrialsPerLineInStage[2];
	nLinesToFitInStage[0] = (int)floor(nLinesToFitInStage1);
	nLinesToFitInStage[1] = (int)floor(nLinesToFitInStage2);
	nTrialsPerLineInStage[0] = (int)floor(nTrialsPerLineInStage1);
	nTrialsPerLineInStage[1] = (int)floor(nTrialsPerLineInStage2);

	LFLineFitter lf;
	lf.Configure(sigma_fit_a_line,sigma_find_support,max_gap,nLayer,nLinesToFitInStage,nTrialsPerLineInStage);
	lf.Init();
	lf.FitLine(&inputImage);

	std::cout<<lf.rNLineSegments()<<std::endl;


	//TODO: allow to choose these parameters in a seperate window maybe


	double nDirection = 60;
	double directionCost = 0.5;
	double maxCost = 30;
	double matchingScale = 1.0;
	double dbScale = 0.6761;
	double baseSearchScale = 1.20;
	double minSearchScale = -7;
	double maxSearchScale = 0;
	double baseSearchAspect = 1.1;
	double minSearchAspect = -1;
	double maxSearchAspect = 1;
	double searchStepSize = 2;
	double searchBoundarySize = 2;
	double minCostRatio = 1.0;



	double maxThreshold = 0.12;
	LMLineMatcher lm;

	// Load configuration
	lm.nDirections_ = (int)(nDirection);
	lm.directionCost_ = (float)(directionCost);
	lm.maxCost_ = maxCost;
	lm.scale_ = matchingScale;
	lm.db_scale_ = dbScale;
	lm.baseSearchScale_ = baseSearchScale;
	lm.minSearchScale_ = (int)(minSearchScale);
	lm.maxSearchScale_ = (int)(maxSearchScale);
	lm.baseSearchAspect_ = baseSearchAspect;
	lm.minSearchAspect_ = (int)(minSearchAspect);
	lm.maxSearchAspect_ = (int)(maxSearchAspect);
	lm.searchStepSize_ = (int)(searchStepSize);
	lm.searchBoundarySize_ = (int)(searchBoundarySize);
	lm.minCostRatio_ = minCostRatio;

	//TODO: get the following shit :D
	// Load tempalte files
	// The following codes replace the functionality of the code
	// lm.Init(templateFileName.c_str());


	//TODO: see if you will delete the following or not. These are 1 according to the demo
	//int m = templateImg.rows;
	//int n = templateImg.cols;
	//int nTemplate = max(m,n);
	int nTemplate = 1;//because from the demo it is 1
	lm.ndbImages_ = nTemplate;
	lm.dbImages_ = new EIEdgeImage[lm.ndbImages_];

	std::cout<<"configuration complete."<<std::endl;

	//mxArray *tempShape;
	for(int i=0;i<nTemplate;i++)
	{
		//tempShape = mxGetCell(prhs[1],i);
		//double *pTempShape = lineRep;

		//int nRow = numberOfLineSegments;
		//int nRow = mxGetM(tempShape);
		//int nCol = mxGetN(tempShape);
		//mexPrintf("( m , n ) = ( %d , %d )\n",nRow,nCol);
		lm.dbImages_[i].SetNumDirections(lm.nDirections_);
		lm.dbImages_[i].Read( lineRep, numberOfLineSegments );
		lm.dbImages_[i].Scale(lm.scale_*lm.db_scale_);
	}

	std::cout<<"detection will start..."<<std::endl;

	vector< vector<LMDetWind> > detWindArrays;
	detWindArrays.clear();
	lm.SingleShapeDetectionWithVaryingQuerySize(lf,maxThreshold,detWindArrays);

	int last = detWindArrays.size()-1;
	int nDetWindows = detWindArrays[last].size();

	//mexPrintf("\n\n Matching \n");
	//mexPrintf("Num of Template = %d; ",nTemplate);
	//mexPrintf("Threshold = %lf; nDetWindows = %d\n",*maxThreshold,nDetWindows);

	cv::Mat detWinds(nDetWindows,6, CV_64F);
	cv::Rect roi;
	int currMaxScore = 0;//will hold the max number of detection windows covered by the current detection window. From github description
	for(int i=0;i<nDetWindows;i++)
	{
		detWinds.at<double>(cv::Point(0,i)) = 1.0*detWindArrays[last][i].x_;
		detWinds.at<double>(cv::Point(1,i)) = 1.0*detWindArrays[last][i].y_;
		detWinds.at<double>(cv::Point(2,i)) = 1.0*detWindArrays[last][i].width_;
		detWinds.at<double>(cv::Point(3,i)) = 1.0*detWindArrays[last][i].height_;
		detWinds.at<double>(cv::Point(4,i)) = 1.0*detWindArrays[last][i].cost_;
		detWinds.at<double>(cv::Point(5,i)) = 1.0*detWindArrays[last][i].count_;

		if (detWindArrays[last][i].count_ > currMaxScore)
		{
			currMaxScore = detWindArrays[last][i].count_;
			roi.x = detWindArrays[last][i].x_;
			roi.y = detWindArrays[last][i].y_;
			roi.width = detWindArrays[last][i].width_;
			roi.height = detWindArrays[last][i].height_;

		}
	}
	return roi;

}


//TODO: change name to lineRep
/*
 * The following takes the edge map of the template image as an input, and will return the line representation in the second parameter
 * This function will then return the number of line segments that represent the input image.
 */
int LineRepresentation(cv::Mat inputImg)
{
	// Create Iamge
	Image<uchar> inputImage;
	inputImage.Resize(inputImg.cols,inputImg.rows,false);

	size_t index1,index2;
	index2 = 0;
	int    row,col;
	for (col=0; col < inputImg.cols; col++)
	{
		for (row=0; row < inputImg.rows; row++)
		{
			inputImage.Access(col,row) = inputImg.at<uchar>(cv::Point(col,row));
		}
	}

	double sigma_fit_a_line = 0.5;
	double sigma_find_support = 0.5;
	double max_gap = 2.0;
	double nLinesToFitInStage1 = 300;
	double nTrialsPerLineInStage1 = 100;
	double nLinesToFitInStage2 = 100000;
	double nTrialsPerLineInStage2 = 1;

	//TODO: allow to choose these parameters in a seperate window maybe

	int nLayer = 2;
	int nLinesToFitInStage[2];
	int nTrialsPerLineInStage[2];
	nLinesToFitInStage[0] = (int)floor(nLinesToFitInStage1);
	nLinesToFitInStage[1] = (int)floor(nLinesToFitInStage2);
	nTrialsPerLineInStage[0] = (int)floor(nTrialsPerLineInStage1);
	nTrialsPerLineInStage[1] = (int)floor(nTrialsPerLineInStage2);

	LFLineFitter lf;
	lf.Configure(sigma_fit_a_line,sigma_find_support,max_gap,nLayer,nLinesToFitInStage,nTrialsPerLineInStage);
	lf.Init();
	lf.FitLine(&inputImage);

	int numberOfLineSegments = lf.rNLineSegments();
	lineRep = new double [numberOfLineSegments * 4];
	//cv::Mat lineRep(numberOfLineSegments,4, CV_64F);
	for(int i = 0; i < numberOfLineSegments; i++)
	{
		lineRep[i+0*numberOfLineSegments] = lf.outEdgeMap_[i].sx_;
		lineRep[i+1*numberOfLineSegments] = lf.outEdgeMap_[i].sy_;
		lineRep[i+2*numberOfLineSegments] = lf.outEdgeMap_[i].ex_;
		lineRep[i+3*numberOfLineSegments] = lf.outEdgeMap_[i].ey_;
	}

	/*
	Image<uchar> *debugImage = lf.ComputeOuputLineImage(&inputImage);

	cv::Mat outpuImg(inputImg.size(), inputImg.type());

	for (col=0; col < outpuImg.cols; col++)
	{
		for (row=0; row < outpuImg.rows; row++)
		{
			outpuImg.at<uchar>(cv::Point(col,row)) = debugImage->Access(col,row);
		}
	}

	delete debugImage;

	*/

	return numberOfLineSegments;

}

cv::Mat CannyEdgeDetector(cv::Mat inputImg, int lowThreshold = 50, int kernel_size = 3)
{
	cv::Mat dst, detected_edges;

	int ratio = 3;
	dst.create( inputImg.size(), inputImg.type() );
	blur( inputImg, detected_edges, cv::Size(kernel_size,kernel_size) );

	//Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
	Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio);
	dst = cv::Scalar::all(0);
	inputImg.copyTo( dst, detected_edges);
	return dst;
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500 )
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size(), 0, 0, CV_INTER_AREA );

    return square;
}

int Run(cv::Mat templateImg, cv::Mat queryImg)
{

// Basic Configuration
//==================================================================

/*
	Configurations are moved into their intended functions
*/
	//cv::Mat	inputImg = CannyEdgeDetector(templateImg);
	int dim = 500;

//==================================================================
// Convert edge map into line representation
//==================================================================

	//cv::Mat tempEdgeMap = templateImg;
	cv::Mat tempEdgeMap = CannyEdgeDetector(templateImg);
	//cv::Mat tempEdgeMap = CannyEdgeDetector(GetSquareImage(templateImg, dim), 50, 3);
	cv::imwrite("Output/tempEdgeMap.jpg", tempEdgeMap);
	int numberOfLineSegments = LineRepresentation(tempEdgeMap);
	std::cout<<numberOfLineSegments<<std::endl;

//==================================================================
// FDCM detection
//==================================================================

	//cv::Mat queryEdgeMap = queryImg;
	cv::Mat queryEdgeMap = CannyEdgeDetector(queryImg);
	//cv::Mat queryEdgeMap = CannyEdgeDetector(GetSquareImage(queryImg, dim), 50, 3);
	cv::imwrite("Output/queryEdgeMap.jpg", queryEdgeMap);
	cv::Rect roi = FdcmDetection(queryEdgeMap, numberOfLineSegments);


	rectangle(queryImg, roi, cv::Scalar( 255 ), 3, 8);
	cv::imwrite("Output/result.jpg", queryImg);

//-------------------------------------------------------
	delete lineRep;

    return EXIT_SUCCESS;
}

int main( int argc, char** argv )
{
	ros::init(argc, argv, "chamfer_matching");
	cv::String templateName, query_location;
	if (argc > 2)
	{

		templateName = argv[1];
		query_location = argv[2];
		if (argc > 3)
		{
			printf("Too many arguments.\n\nPlease enter:\n\t1. The location of the template image\n\t2. The location of the query image\n\n");
			return 1;
		}
	}
	else
	{
		printf("Too few arguments.\n\nPlease enter:\n\t1. The location of the template image\n\t2. The location of the query image\n\n");
		return 1;
	}

	cv::Mat templateImg;
	templateImg = imread(templateName, CV_LOAD_IMAGE_GRAYSCALE);
	//resize(imread(templateName, CV_LOAD_IMAGE_GRAYSCALE), templateImg, cv::Size(640, 512), 0, 0, cv::INTER_AREA);
	if(templateImg.empty())
	{
		std::cout << "Couldn't load " << templateName << std::endl;
		//cmd.printMessage();
		printf("Wrong input arguments.\n\nPlease enter:\n\t1. The location of the template image\n\t2. The location of the query image\n\n");
		return EXIT_FAILURE;
	}

	cv::Mat queryImg;
	queryImg = imread(query_location, CV_LOAD_IMAGE_GRAYSCALE);
	//resize(imread(query_location, CV_LOAD_IMAGE_GRAYSCALE), queryImg, cv::Size(640, 512), 0, 0, cv::INTER_AREA);
	if(queryImg.empty())
	{
		std::cout << "Couldn't load " << query_location << std::endl;
		//cmd.printMessage();
		printf("Wrong input arguments.\n\nPlease enter:\n\t1. The location of the template image\n\t2. The location of the query image\n\n");
		return EXIT_FAILURE;
	}

 	return Run(templateImg, queryImg);


}

