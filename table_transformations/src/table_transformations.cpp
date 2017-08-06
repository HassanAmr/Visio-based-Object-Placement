#include <ros/ros.h>
#include <image_transport/image_transport.h>
//#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
#include <stdio.h>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//using namespace cv;

image_transport::Subscriber sub;
int markerBoxSize = 30;

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

int Run(cv::String folderpath, cv::String outputPath, bool folderSource, cv::Mat subImage)
{
    std::vector<cv::String> filenames;
    int mainLoopCount;
    if (folderSource)
    {
    	glob(folderpath, filenames);

    	std::cout <<filenames.size()<< std::endl;
    	mainLoopCount  = filenames.size();
    }
    else
    {
    	mainLoopCount = 1;
    }

    bool patternfound = false;
    int l,k;
    //#pragma omp parallel default(shared) private(l, k)

    cv::String hMatrices_path = outputPath + "matrices/H_Matrices.xml";
    std::cout << hMatrices_path << std::endl;

    cv::FileStorage hMatrices(hMatrices_path, cv::FileStorage::WRITE);
    //#pragma omp parallel for private(patternfound, corners)
    cv::Size patternsize;//this is just a dummy declaration because it is only declared inside a loop, and the compiler doesn't like it

    for (size_t i=0; i< mainLoopCount; i++)
    {
    	cv::Mat im;
    	cv::String fileNameStr;
    	if (folderSource)
    	{
    		if (SplitFilename(filenames[i])[0] == '.')
    		{
    			continue;
    		}


    		im = imread(filenames[i]);
            fileNameStr = SplitFilename(filenames[i]);
    	}
    	else
    	{
    		subImage.copyTo(im);

    		fileNameStr = "output_image.jpg";
    	}
        std::vector<cv::Point2f> corners; //this will be filled by the detected corners

        l = 7;
        while (!patternfound && l > 2)
        {
          k = 7;
          while (!patternfound && k > 2)
          {
            patternsize.height = l;
            patternsize.width = k;
            patternfound = findChessboardCorners(im, patternsize, corners,cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
            k--;
          }
          l--;
        }

        if (!patternfound)
        {
          continue;
        }
        int x = 0;
        int y = 0;

        std::vector<cv::Point2f> marker;
        std::vector<cv::Point3f> patternPoints3d;

        int imageCenterX = im.cols/2;
        int imageCenterY = im.rows/2;
        //int c = 0;
        for (int  h = 0; h < patternsize.height; ++h)
        {
            for (int w = 0; w < patternsize.width; ++w)
            {
                cv::Point2f p2(w*markerBoxSize,h*markerBoxSize);
                marker.push_back(p2);
                //cv::Point3f p3((corners[c].x - imageCenterX), (corners[c].y - imageCenterY), 0);
                cv::Point3f p3(w, h, 0);
                patternPoints3d.push_back(p3);
                //c++;
            }
        }
        //std::cout<< c << std::endl;
        //std::cout<< patternPoints3d<< std::endl;
        cv::Mat dst;
        resize(im, dst, im.size(), 0, 0, cv::INTER_AREA); // resize to 640x512 resolution

        //std::cout<<patternsize<<std::endl;

        std::ostringstream ss;
        ss <<outputPath << "images/"<< fileNameStr;
        cv::String outputName = ss.str();

        std::cout <<outputName<< std::endl;

        imwrite(outputName , dst);

        drawChessboardCorners(dst, patternsize, corners, patternfound);
        drawChessboardCorners(dst, patternsize, marker, patternfound);
        patternfound = false;

        std::ostringstream ss2;
        ss2 <<outputPath << "annotation/"<< fileNameStr;

        cv::String outputName2 = ss2.str();
        //std::cout <<patternsize.height<< std::endl;
        //std::cout <<patternsize.width<< std::endl;


        std::vector<cv::Point2f> obj_corners(4);
        std::vector<cv::Point2f> scene_corners(4);


        obj_corners[0] = cv::Point(0,0);
        obj_corners[1] = cv::Point( (patternsize.width - 1) * markerBoxSize, 0 );
        obj_corners[2] = cv::Point( (patternsize.width - 1) * markerBoxSize, (patternsize.height - 1) * markerBoxSize );
        obj_corners[3] = cv::Point( 0, (patternsize.height - 1) * markerBoxSize );

        cv::Mat H = findHomography( marker, corners, cv::RANSAC );
        perspectiveTransform( obj_corners, scene_corners, H);


        //std::vector<cv::Point3f> patternPoints3d;
        //for (int  h = 0; h < patternsize.height; ++h)
        //{
        //  for (int w = 0; w < patternsize.width; ++w)
        //  {
        //      cv::Point3f p3(w*markerBoxSize,h*markerBoxSize, 0);
        //      patternPoints3d.push_back(p3);

        //  }
        //}

        std::vector<std::vector<cv::Point3f> > objectPoints;
        std::vector<std::vector<cv::Point2f> > imagePoints;
        std::vector<cv::Point3f> axis;
        axis.push_back(cv::Point3f(3, 0, 0));
        axis.push_back(cv::Point3f(0, 3, 0));
        axis.push_back(cv::Point3f(0, 0, -3));


        //cv::Mat cameraMatrix, distMatrix;
        std::vector<cv::Mat> _rvecs, _tvecs;
        cv::Mat rvecs = (cv::Mat_<double>(3, 1) << 0.08257, -0.6168, 1.4675);
        cv::Mat tvecs = (cv::Mat_<double>(3, 1) << -0.3806, -0.1605, 0.6087);

        cv::Mat CamMatrix = cv::Mat::eye(3, 3, CV_32F);
        CamMatrix.at< float >(0, 0) = 500;
        CamMatrix.at< float >(1, 1) = 500;
        CamMatrix.at< float >(0, 2) = imageCenterX;
        CamMatrix.at< float >(1, 2) = imageCenterY;
        cv::Mat distMatrix= cv::Mat::zeros(1, 5, CV_32F);

        //solvePnPRansac(patternPoints3d, corners, CamMatrix, distMatrix, rvecs, tvecs);

        //_rvecs.push_back(rvecs);
        //_tvecs.push_back(tvecs);
        objectPoints.push_back(patternPoints3d);
        imagePoints.push_back(corners);

        calibrateCamera(objectPoints, imagePoints, im.size(), CamMatrix, distMatrix, _rvecs, _tvecs);

        solvePnPRansac(patternPoints3d, corners, CamMatrix, distMatrix, rvecs, tvecs);
        //std::cout<<rvecs<<std::endl;

        std::vector<cv::Point2f>  projectedPoints;

        projectPoints(axis, rvecs, tvecs, CamMatrix, distMatrix, projectedPoints);

        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        line( dst,
                scene_corners[0], scene_corners[1],
            cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );
        line( dst,
                scene_corners[1], scene_corners[2],
            cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );
        line( dst,
                scene_corners[2], scene_corners[3],
            cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );
        line( dst,
                scene_corners[3], scene_corners[0],
            cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );


        line( dst,
                scene_corners[0], projectedPoints[0],
            cv::Scalar( 255, 0, 0), 2, cv::LINE_AA );
        line( dst,
                scene_corners[0], projectedPoints[1],
            cv::Scalar( 255, 0, 255), 2, cv::LINE_AA );
        line( dst,
                scene_corners[0], projectedPoints[2],
            cv::Scalar( 0, 0, 255), 2, cv::LINE_AA );

        imwrite(outputName2 , dst);


        //Camera parameters and matrices should be cleared since every picture should not be related to the rest of the pictures
        hMatrices << RemoveFileExtension(fileNameStr).c_str() << H;
        obj_corners.clear();
        scene_corners.clear();
        H.release();
        //break;
    }
    hMatrices.release();
    return EXIT_SUCCESS;
}

/*void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	std::cout <<"here2" <<std::endl;
	try
	{
		subImage = cv_bridge::toCvShare(msg, "bgr8")->image;
		sub.shutdown();
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}*/
int main( int argc, char** argv )
{
	cv::String folderpath, outputPath;

	if (argc > 2)
	{
		folderpath = argv[1];
		outputPath = argv[2];

		if (argc > 3)
		{
			printf("Too many arguments. Please enter the path of the files you wish to sample, then followed by the location in which you want to have the new samples.\n\n");
			return 1;
		}

	}
	else
	{
		printf("Too few arguments. Please enter the path of the files you wish to sample, then followed by the location in which you want to have the new samples.\n\n");
		return 1;
	}
	bool folderSource = true;
	if (folderpath == "--sub")
	{
		folderSource = false;
		ros::init(argc, argv, "table_transormations");
		//ros::NodeHandle nh;
		//image_transport::ImageTransport it(nh);
		//sub = it.subscribe("/input_image", 1, imageCallback);
    	//std::cout <<"here1" <<std::endl;
    	//ros::spin();
		//ros::spinOnce();
		sensor_msgs::ImageConstPtr msg = ros::topic::waitForMessage<sensor_msgs::Image>("/input_image");

		try
		{
			cv::Mat subImage = cv_bridge::toCvShare(msg, "bgr8")->image;
			return Run(folderpath, outputPath, folderSource, subImage);


		}
		catch (cv_bridge::Exception& e)
		{
			ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
		}
	}
	else
	{
		cv::Mat subImage;
		return Run(folderpath, outputPath, folderSource, subImage);

	}

}

