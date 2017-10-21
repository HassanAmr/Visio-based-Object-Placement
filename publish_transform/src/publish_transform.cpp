#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <fstream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/imgcodecs.hpp"
//#include "opencv2/imgproc.hpp"
//#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/video.hpp>
#include "opencv2/core/eigen.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <MoveItController.h>

cv::String query_location, bgs_location, test_location, cwd, output_location, log_location, homographyMethod;//cwd is short for curring working directory


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


int main(int argc, char **argv)
{

	ros::init(argc, argv, "publish_transform");

	std::string inputParam;
	ros::NodeHandle nh;

	while (!(nh.getParam("/visiobased_placement/cwd", inputParam)) && nh.ok())
	{}
	cwd = inputParam;
	std::cout<<inputParam<<std::endl;
	inputParam = "";

	if (!(nh.getParam("/visiobased_placement/CACHED_QUERY_FILE_NAME", inputParam)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	query_location = inputParam;
	inputParam = "";

	if (!(nh.getParam("/visiobased_placement/CACHED_BGS_FILE_NAME", inputParam)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	bgs_location = inputParam;
	inputParam = "";

	if (!(nh.getParam("/visiobased_placement/UPRIGHT_PATH", inputParam)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	test_location = inputParam;
	inputParam = "";


	if (!(nh.getParam("/visiobased_placement/ROTATION_PATH", inputParam)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	output_location = inputParam;


	if (!(nh.getParam("/visiobased_placement/LOG_PATH", inputParam)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	log_location = inputParam;


	int focal_lenth = 500;
	if (!(nh.getParam("/visiobased_placement/CAMERA_FL", focal_lenth)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	query_location = cwd + query_location;
	bgs_location = cwd+ bgs_location;
	test_location = cwd + test_location;
	output_location = cwd + output_location;
	log_location = cwd + log_location;

	/*
	cv::UMat queryImg;
	queryImg = imread(bgs_location, CV_LOAD_IMAGE_GRAYSCALE).getUMat( cv::ACCESS_READ );
	if(queryImg.empty())
	{
		std::cout << "Couldn't load " << query_location << std::endl;
		//cmd.printMessage();
		printf("Something went wrong loading the background subtractged image.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}
	*/

	//cv::UMat img1;
	//queryImg.copyTo(img1);

	//inputs
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector<cv::DMatch>  matches;
	cv::Mat H;
	int bestImageWidth, bestImageHeight;

	cv::String inName= log_location + "/transformationOutput.yml";
	cv::FileStorage inTF(inName, cv::FileStorage::READ);

	//inTF["fileName"] >> bestFileName.c_str();
	//inTF["fitnessScore"] >> maxFitnessScore;
	//inTF["minDist"] >> bestMinDist;
	inTF["homographyMatrix"] >> H;
	//inTF["rotationMatrix"] >> oRvecs[selectedR];
	inTF["imageWidth"] >> bestImageWidth;
	inTF["imageHeight"] >> bestImageHeight;
	inTF["keypoints1"] >> keypoints1;
	inTF["keypoints2"] >> keypoints2;
	inTF["matches"] >> matches;
	inTF.release();
    //compute transform matrix then print
	std::vector<cv::Mat> oRvecs, oTvecs, oNvecs;

	cv::Mat CamMatrix = cv::Mat::eye(3, 3, CV_32F);

	CamMatrix.at< float >(0, 0) = focal_lenth;
	CamMatrix.at< float >(1, 1) = focal_lenth;
	CamMatrix.at< float >(0, 2) = bestImageWidth/2;
	CamMatrix.at< float >(1, 2) = bestImageHeight/2;

	decomposeHomographyMat(H, CamMatrix, oRvecs, oTvecs, oNvecs);

	std::cout <<"Rotation matrices acquired:" << std::endl;
	double diff = 1;
	int selectedR = -1;//this is to force an error if no one was chosen
	for (int  j = 0; j < oRvecs.size(); ++j)
	{
		std::cout <<oRvecs[j]<< std::endl;
		//std::cout <<oRvecs[j].at<double>(2,2)<< std::endl;
		double newDiff = 1 - oRvecs[j].at<double>(2,2);
		if (newDiff < diff)
		{
			diff = newDiff;
			selectedR = j;
		}
	}


	//---------------------------------------------------------------------------------------------------

	//TODO: Fix with Daniel
	//MoveItController *m_armController;
	//m_armController = new MoveItController("iiwa", SCHUNK_HAND);
	//m_armController->init(nh);
	//m_armController->closeGripper();

	//controller->init(nh);
	//Eigen::Vector3f currPos = controller->getCurrentPosition();
	Eigen::Matrix3f orientation;
	cv2eigen(oRvecs[selectedR], orientation);

	//controller->goToPosition(currPos, orientation);


	std::cout <<"Publishing transform..." << std::endl;
	tf::TransformBroadcaster * br;
	br = new tf::TransformBroadcaster;
	ros::Rate r(10);
	while(ros::ok()){

		Eigen::Quaternionf q(orientation);
		cv::String tfStr = "rvec_frame";
		tf::StampedTransform transform;
		tf::Quaternion t;
		tf::quaternionEigenToTF(Eigen::Quaterniond(q), t);
		//tf::()
		transform.setRotation(t);
		br->sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/kinect2_ir_optical_frame", tfStr));

		r.sleep();
	}

	return EXIT_SUCCESS;
}
