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


#include <actionlib/client/simple_action_client.h>
#include <eigen_conversions/eigen_msg.h>
#include <prm_planner_msgs/GoalAction.h>
#include <Eigen/Geometry>


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
	ros::init(argc, argv, "prm_planner_action_server_interface");
	ros::NodeHandle n;

	//create action client and wait until server is available
	actionlib::SimpleActionClient<prm_planner_msgs::GoalAction> actionClient("prm_planner/goals", true);
	actionClient.waitForServer();

	//define goals: We use the end effector pose relative to the planning frame as a goal

	ros::AsyncSpinner spinner(2);
	spinner.start();

	tf::StampedTransform transform, transform1, transform2;
	tf::TransformListener listener;

	while (ros::ok())
	{
		sleep(1);

		try{
			listener.lookupTransform("/iiwa/iiwa_0_link", "/object_at_hand_1", ros::Time(0), transform1);
			listener.lookupTransform("/iiwa/iiwa_0_link", "/object_at_hand_2", ros::Time(0), transform2);

		}
		catch (tf::TransformException &ex) {
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
		}

		Eigen::Affine3d goal1;
		Eigen::Affine3d goal2;
		const tf::Transform tf1(transform1.getBasis(),transform1.getOrigin());
		const tf::Transform tf2(transform2.getBasis(),transform2.getOrigin());

		tf::transformTFToEigen (tf1, goal1);
		tf::transformTFToEigen (tf2, goal2);

		prm_planner_msgs::GoalGoal goalMsg1;
		tf::poseEigenToMsg(goal1, goalMsg1.goal);
		goalMsg1.action = prm_planner_msgs::GoalGoal::ACTION_MOVE;

		prm_planner_msgs::GoalGoal goalMsg2;
		tf::poseEigenToMsg(goal2, goalMsg2.goal);
		goalMsg2.action = prm_planner_msgs::GoalGoal::ACTION_MOVE;



		actionClient.sendGoal(goalMsg1);
		actionClient.waitForResult();
		if (!actionClient.getResult()->success)
		{
			ROS_ERROR("Cannot find a trajectory to goal 1");
		}

		sleep(5);

		if (!ros::ok())
			break;

		actionClient.sendGoal(goalMsg2);
		actionClient.waitForResult();
		if (!actionClient.getResult()->success)
		{
			ROS_ERROR("Cannot find a trajectory to goal 2");
		}

		sleep(1);
	}

	return EXIT_SUCCESS;
/*
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


	double diff = 1;
	int selectedR1 = 0, selectedR2 = 1;
	if (oRvecs[selectedR1].at<double>(2,2) == oRvecs[selectedR2].at<double>(2,2))
		selectedR2++;

	std::cout <<"Rotation matrices acquired:" << std::endl;
	std::cout <<oRvecs[selectedR1]<< std::endl;
	std::cout <<oRvecs[selectedR2]<< std::endl;


	Eigen::Matrix3f orientation1, orientation2;
	cv2eigen(oRvecs[selectedR1], orientation1);
	cv2eigen(oRvecs[selectedR2], orientation2);

	//controller->goToPosition(currPos, orientation);


	std::cout <<"Publishing transform..." << std::endl;
	tf::TransformBroadcaster * br;
	br = new tf::TransformBroadcaster;
	ros::Rate r(10);
	while(ros::ok()){

		Eigen::Quaternionf q1(orientation1);
		Eigen::Quaternionf q2(orientation2);

		cv::String tfStr1 = "r1_frame";
		cv::String tfStr2 = "r2_frame";
		tf::StampedTransform transform1;
		tf::StampedTransform transform2;
		tf::Quaternion t1;
		tf::Quaternion t2;
		tf::quaternionEigenToTF(Eigen::Quaterniond(q1), t1);
		tf::quaternionEigenToTF(Eigen::Quaterniond(q2), t2);
		//tf::()
		transform1.setRotation(t1);
		transform2.setRotation(t2);
		br->sendTransform(tf::StampedTransform(transform1, ros::Time::now(), "/kinect2_ir_optical_frame", tfStr1));
		br->sendTransform(tf::StampedTransform(transform2, ros::Time::now(), "/kinect2_ir_optical_frame", tfStr2));
		r.sleep();
	}

	return EXIT_SUCCESS;
	*/
}
