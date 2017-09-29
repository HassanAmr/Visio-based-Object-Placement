#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <cv_bridge/cv_bridge.h>

#include <iostream>
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


int outputCounter = 1;
const float ratio_Lowe = 0.8f; // As in Lowe's paper; can be tuned


const int GOOD_PORTION = 10;

//junk delete later
//cv::FileStorage f_verification("output/verification/F_verification.xml", cv::FileStorage::WRITE);
//cv::FileStorage e_verification("output/verification/E_verification.xml", cv::FileStorage::WRITE);

struct SURFDetector
{
	cv::Ptr<cv::Feature2D> surf;
    SURFDetector(double hessian = 800.0)
    {
        surf = cv::xfeatures2d::SURF::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};

struct SIFTDetector
{
	cv::Ptr<cv::Feature2D> sift;
    SIFTDetector(double hessian = 800.0)
    {
        sift = cv::xfeatures2d::SIFT::create(hessian);
    }
    template<class T>
    void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
    {
        sift->detectAndCompute(in, mask, pts, descriptors, useProvided);
    }
};
/*
template<class KPMatcher>
struct SURFMatcher
{
    KPMatcher matcher;
    template<class T>
    void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
    {
        matcher.match(in1, in2, matches);
    }
};
*/
struct FirstColumnOnlyCmp
{
    bool operator()(const std::vector<int>& lhs,
                    const std::vector<int>& rhs) const
    {
        return lhs[1] > rhs[1];
    }
};


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

static cv::Mat drawGoodMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& img1,
    const cv::Mat& img2,
    std::vector<cv::DMatch>& good_matches
    )
{
    // drawing the results
	cv::Mat img_matches;

    drawMatches( img1, keypoints1, img2, keypoints2,
                 good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::DEFAULT  );
    //return img_matches;

    //-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }



    //-- Get the corners from the image_1 ( the object to be "detected" )
    int max_x = 0;
    int min_x = 5000;//works as infinity for such image sizes
    int max_y = 0;
    int min_y = 5000;//works as infinity for such image sizes
    int curX, curY;
    for (size_t i = 0; i < keypoints1.size(); i++)
    {
        curX = keypoints1[i].pt.x;
        curY = keypoints1[i].pt.y;
        if (curX > max_x)
            max_x = curX;
        if (curY > max_y)
            max_y = curY;

        if (curX < min_x)
            min_x = curX;
        if (curY < min_y)
            min_y = curY;
    }

    std::vector<cv::Point2f> obj_corners(4);
    obj_corners[0] = cv::Point(min_x,min_y);
    obj_corners[1] = cv::Point( max_x, min_y );
    obj_corners[2] = cv::Point( max_x, max_y );
    obj_corners[3] = cv::Point( min_x, max_y );
    std::vector<cv::Point2f> scene_corners(4);


    //obj_corners[0] = Point(0,0);
    //obj_corners[1] = Point( cols, 0 );
    //obj_corners[2] = Point( cols, rows );
    //obj_corners[3] = Point( 0, rows );
    //TODO: FIX

    cv::Mat H = findHomography( obj, scene, cv::RANSAC );
    perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
    line( img_matches,
          scene_corners[0] + cv::Point2f( (float)img1.cols, 0), scene_corners[1] + cv::Point2f( (float)img1.cols, 0),
		  cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );
    line( img_matches,
          scene_corners[1] + cv::Point2f( (float)img1.cols, 0), scene_corners[2] + cv::Point2f( (float)img1.cols, 0),
		  cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );
    line( img_matches,
          scene_corners[2] + cv::Point2f( (float)img1.cols, 0), scene_corners[3] + cv::Point2f( (float)img1.cols, 0),
		  cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );
    line( img_matches,
          scene_corners[3] + cv::Point2f( (float)img1.cols, 0), scene_corners[0] + cv::Point2f( (float)img1.cols, 0),
		  cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );

    return img_matches;
}

//later should be void again, and Mat should go to the new function that will be specifically responsible for homography
cv::Mat findGoodMatches(
    int cols, int rows,//the columns and rows that cover exactly how big the object is, used for RANSAC homography
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    std::vector< std::vector<cv::DMatch> >& matches,
    std::vector<cv::DMatch>& backward_matches,
    std::vector<cv::DMatch>& selected_matches,
    std::string currImgText
    )
{
    //-- Sort matches and preserve top 10% matches
    std::sort(matches.begin(), matches.end());
    std::vector< cv::DMatch > good_matches;
    double minDist = matches.front()[0].distance;
    double maxDist = matches.back()[0].distance;

    for( int i = 0; i < matches.size(); i++ )
    {
        if (matches[i][0].distance < ratio_Lowe * matches[i][1].distance)
        {
        	cv::DMatch forward = matches[i][0];
        	cv::DMatch backward = backward_matches[forward.trainIdx];
            if(backward.trainIdx == forward.queryIdx)
            {
                good_matches.push_back(forward);
            }

        }
        //good_matches.push_back( matches[i][0] );
        //good_matches.push_back( matches[i] );
    }
    //std::cout<<good_matches.size() << "    ";
    //std::cout << "\nMax distance: " << maxDist << std::endl;
    //std::cout << "Min distance: " << minDist << std::endl;

    //std::cout << "Calculating homography using " << ptsPairs << " point pairs." << std::endl;

//-- Localize the object
    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
    }


    std::vector<cv::Point2f> obj_corners(4);
    std::vector<cv::Point2f> scene_corners(4);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat H;//, M; //The matrices to be returned
    if (obj.size() > 0)
    {
        //-- Get the corners from the image_1 ( the object to be "detected" )
        int max_x = 0;
        int min_x = 5000;//works as infinity for such image sizes
        int max_y = 0;
        int min_y = 5000;//works as infinity for such image sizes
        int curX, curY;
        for (size_t i = 0; i < keypoints1.size(); i++)
        {
            curX = keypoints1[i].pt.x;
            curY = keypoints1[i].pt.y;
            if (curX > max_x)
                max_x = curX;
            if (curY > max_y)
                max_y = curY;

            if (curX < min_x)
                min_x = curX;
            if (curY < min_y)
                min_y = curY;
        }

        obj_corners[0] = cv::Point(min_x,min_y);
        obj_corners[1] = cv::Point( max_x, min_y );
        obj_corners[2] = cv::Point( max_x, max_y );
        obj_corners[3] = cv::Point( min_x, max_y );


        //obj_corners[0] = Point(0,0);
        //obj_corners[1] = Point( cols, 0 );
        //obj_corners[2] = Point( cols, rows );
        //obj_corners[3] = Point( 0, rows );

        H = findHomography( obj, scene, cv::RANSAC, 3 );

        //std::cout << H << ": Not a proper match. " << good_matches.size() << std::endl;

        if (countNonZero(H) < 1)
        {
            //std::cout << outputCounter++<< ": Not a proper match. " << selected_matches.size() << std::endl;
        }
        else
        {
            perspectiveTransform( obj_corners, scene_corners, H);

            //find out later what this does
            //scene_corners_ = scene_corners;

            //Mat drawing = Mat::zeros( img2.size(), img2.type() );
            //using searchImg since img2 is currently not available, and both are the same size.
            //later should be set to the region where the object surely is.
            cv::Mat drawing = cv::Mat::zeros( cols, rows, CV_8UC1);

            line( drawing,
                scene_corners[0], scene_corners[1],
				cv::Scalar( 255 ), 3, 8 );
            line( drawing,
                scene_corners[1], scene_corners[2],
				cv::Scalar( 255 ), 3, 8 );
            line( drawing,
                scene_corners[2], scene_corners[3],
				cv::Scalar( 255 ), 3, 8 );
            line( drawing,
                scene_corners[3], scene_corners[0],
				cv::Scalar( 255 ), 3, 8 );

            //find contours of the above drawn region
            findContours( drawing, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0) );
            double val = 0.0;

            if (contours.size() > 0)
            {
                for( size_t i = 0; i < good_matches.size(); i++ )
                {
                    val = pointPolygonTest( contours[0], keypoints2[ good_matches[i].trainIdx ].pt , false );

                    if (val >= 0)
                    {
                        selected_matches.push_back(good_matches[i]);
                    }
                }
            }
        }
    }
/*
            if (selected_matches.size() >= 8)
            {
                //-- Get the points corresponding to the selected matches
                std::vector<cv::Point2f> queryPoints;
                std::vector<cv::Point2f> refPoints;

                for( int i = 0; i < selected_matches.size(); i++ )
                {
                    //-- Get the keypoints from the good matches
                    queryPoints.push_back( keypoints1[ selected_matches[i].queryIdx ].pt );
                    refPoints.push_back( keypoints2[ selected_matches[i].trainIdx ].pt );
                }

                cv::Mat F = findFundamentalMat(queryPoints, refPoints, CV_FM_8POINT);

                std::vector<double> verifyValues;
                for (int i = 0; i < selected_matches.size(); i++)
                {
                	cv::Mat queryMatrix(queryPoints[i]);
                	cv::Mat refMatrix(refPoints[i]);
                    queryMatrix.convertTo(queryMatrix,cv::DataType<double>::type);
                    refMatrix.convertTo(refMatrix,cv::DataType<double>::type);
                    cv::Mat one = cv::Mat::ones(1, 1, cv::DataType<double>::type);
                    queryMatrix.push_back(one);
                    refMatrix.push_back (one);

                    cv::Mat final = queryMatrix.t() * F * refMatrix;
                    verifyValues.push_back(final.at<double>(0,0));
                }

                bool ok = false;
                float acc_deviation = 0;
                int cnt = 0;
                for (int i = 0; i < selected_matches.size(); i++)
                {
                    acc_deviation += fabs(verifyValues[i]);
                    if (fabs(verifyValues[i]) < 2)
                    {
                        ok = true;
                        cnt ++;
                    }
                }

                float u_deviation = acc_deviation/selected_matches.size();
                if (ok)
                {
                    f_verification << currImgText<< ("Yes: " + std::to_string(cnt) + " out of " + std::to_string(selected_matches.size()) + " selected matches with a mean deviation of " + std::to_string(u_deviation) + " pixels");
                }
                else
                {
                    f_verification << currImgText<< "No";
                }

                cv::Mat K1 = (cv::Mat_<double>(3,3) << 1076.8879, 0.0, 312.05695, 0.0, 1076.8904, 244.55385, 0.0, 0.0, 1.0);
                //Mat K2 = (Mat_<double>(3,3) << 538.44395, 0.0, 312.05695, 0.0, 538.4452, 244.55385, 0.0, 0.0, 1.0);
                //Mat K1 = (Mat_<double>(3,3) << 538.44395, 0.0, 312.05695, 0.0, 538.4452, 244.55385, 0.0, 0.0, 0.5);
                cv::Mat W = (cv::Mat_<double>(3,3) << 0, -1, 0, 1, 0, 0, 0, 0, 1);

                cv::Mat E = K1.t()* F * K1;
                //K1 E = K1.t()* F * K2;
                //Mat E = K2.t()* F * K1;

                verifyValues.clear();
                for (int i = 0; i < selected_matches.size(); i++)
                {
                	cv::Mat queryMatrix(queryPoints[i]);
                	cv::Mat refMatrix(refPoints[i]);
                    queryMatrix.convertTo(queryMatrix,cv::DataType<double>::type);
                    refMatrix.convertTo(refMatrix,cv::DataType<double>::type);
                    cv::Mat one = cv::Mat::ones(1, 1, cv::DataType<double>::type);
                    queryMatrix.push_back(one);
                    refMatrix.push_back (one);

                    cv::Mat final = queryMatrix.t() * E * refMatrix;
                    verifyValues.push_back(final.at<double>(0,0));
                }

                ok = false;
                acc_deviation = 0;
                cnt = 0;
                for (int i = 0; i < selected_matches.size(); i++)
                {
                    acc_deviation += fabs(verifyValues[i]);
                    if (fabs(verifyValues[i]) < 2)
                    {
                        ok = true;
                        cnt ++;
                    }
                }

                u_deviation = acc_deviation/selected_matches.size();
                if (ok)
                {
                    e_verification << currImgText<< ("Yes: " + std::to_string(cnt) + " out of " + std::to_string(selected_matches.size()) + " selected matches with a mean deviation of " + std::to_string(u_deviation) + " pixels");
                }
                else
                {
                    e_verification << currImgText<< "No";
                }

                cv::SVD svd(E);
                cv::Mat R1 = svd.u *W * svd.vt;
                cv::Mat R2 = svd.u *W.t() * svd.vt;

                //std::cout<< svd.u << std::endl<< svd.w <<std::endl<< svd.u.t()<<std::endl;
                //Mat L = Mat::zeros(3, 3, CV_64F);
                //L.at<double>(0,0) = svd.w.at<double>(0);
                //L.at<double>(1,1) = svd.w.at<double>(1);
                //L.at<double>(2,2) = svd.w.at<double>(2);
                cv::Mat u3(3, 1, cv::DataType<double>::type);
                u3.at<double>(0,0) = svd.u.at<double>(0,2);
                u3.at<double>(1,0) = svd.u.at<double>(1,2);
                u3.at<double>(2,0) = svd.u.at<double>(2,2);
                double min, max;
                minMaxLoc(u3, &min, &max);
                max = std::max(fabs(min),max);
                //std::cout<<svd.u<<std::endl;
                //std::cout<<u3<<std::endl;
                cv::Mat T1(3, 1, cv::DataType<double>::type);
                cv::Mat T2(3, 1, cv::DataType<double>::type);
                T1.at<double>(0,0) = u3.at<double>(0,0)/max;
                T1.at<double>(1,0) = u3.at<double>(1,0)/max;
                T1.at<double>(2,0) = u3.at<double>(2,0)/max;

                T2.at<double>(0,0) = -1 * (u3.at<double>(0,0)/max);
                T2.at<double>(1,0) = -1 * (u3.at<double>(1,0)/max);
                T2.at<double>(2,0) = -1 * (u3.at<double>(2,0)/max);

                //std::cout << L<<std::endl;
                //Mat T = svd.u *L * W * svd.u.t();
                cv::Mat T;
                hconcat(T1,T2, T);
                cv::Mat R;
                hconcat(R1, R2, R);
                cv::Mat transformations;
                hconcat(R, T, transformations);
                cv::Mat FE;
                hconcat(F, E, FE);
                //hconcat(H, T, M);
                cv::Mat misc;
                hconcat(H, FE, misc);
                hconcat(misc, transformations, M);
                return M;
            }
        }
	}

    //f_verification << currImgText<< "No";
    //e_verification << currImgText<< "No";
*/
    return H;
}

////////////////////////////////////////////////////
// This program demonstrates the usage of SURF_OCL.
// use cpu findHomography interface to calculate the transformation matrix
/*
void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
  try
  {
    cv::imshow("view", cv_bridge::toCvShare(msg, "bgr8")->image);
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}
*/

int main(int argc, char **argv)
{

	ros::init(argc, argv, "object_transormations");

	cv::String query_location, test_location, cwd, output_location;//cwd is short for curring working directory
	std::string inputParam;
	ros::NodeHandle nh;

	while (!(nh.getParam("/visiobased_placement/cwd", inputParam)) && nh.ok())
	{}
	cwd = inputParam;
	inputParam = "";

	if (!(nh.getParam("/visiobased_placement/CACHED_QUERY_FILE_NAME", inputParam)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	query_location = inputParam;
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
	//std::cout << cwd << std::endl;
	//std::cout << query_location << std::endl;
	//std::cout << test_location << std::endl;
	//std::cout << output_location << std::endl;

	//TODO: Get the following from the rosparam server
	query_location = cwd + query_location;
	test_location = cwd + test_location;
	output_location = cwd + output_location;
	const cv::String dataset_type = ".jpg"; //TODO: maybe set it from the params.yaml


	cv::UMat queryImg;
	queryImg = imread(query_location, CV_LOAD_IMAGE_GRAYSCALE).getUMat( cv::ACCESS_READ );
	if(queryImg.empty())
	{
		std::cout << "Couldn't load " << query_location << std::endl;
		//cmd.printMessage();
		printf("Something went wrong loading query image.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	cv::UMat queryColorImg;
	imread(query_location, CV_LOAD_IMAGE_COLOR).copyTo(queryColorImg);

	//Prepare for main loop
	std::vector<cv::String> filenames;
	glob(test_location, filenames);

	cv::UMat img1;
	queryImg.copyTo(img1);

	//declare input/output
	std::vector<cv::KeyPoint> keypoints1, keypoints2;
	std::vector< std::vector<cv::DMatch> > matches;
	std::vector<cv::DMatch> backward_matches;

	cv::UMat _descriptors1, _descriptors2;
	cv::Mat descriptors1 = _descriptors1.getMat(cv::ACCESS_RW),
			descriptors2 = _descriptors2.getMat(cv::ACCESS_RW);

	//instantiate detectors/matchers
	//SURFDetector surf;
	SIFTDetector sift;

	//SURFMatcher<BFMatcher> matcher;
	cv::BFMatcher matcher;


	//surf(img1.getMat(cv::ACCESS_READ), cv::Mat(), keypoints1, descriptors1);
	sift(img1.getMat(cv::ACCESS_READ), cv::Mat(), keypoints1, descriptors1);

	std::vector< std::vector<cv::DMatch> > final_matches;

	//std::vector<cv::Mat> allMatrices;

	//store the matrices in a file
	//cv::FileStorage matrices("output/Matrices.xml", cv::FileStorage::WRITE);

	int maxMatches = 0;
	int matchesFound = 0;
	cv::Mat currM;

	//Best result data
	int numOfMatches = 0;
	cv::String bestFileName;
	cv::Mat bestM;
	std::vector<cv::DMatch> bestMatches;
	int bestKeypointsNum = 0;
	int bestImageCenterX = 0;
	int bestImageCenterY = 0;
	cv::Mat H;

	for (size_t i=0; i< filenames.size(); i++)
	{

		cv::UMat img2;
		cv::UMat testImg;
		testImg = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE).getUMat( cv::ACCESS_READ );
		if(queryImg.empty())
		{
			std::cout << "Couldn't load " << test_location << std::endl;
			//cmd.printMessage();
			printf("Something went wrong loading %s.\nSkipping over it.", filenames[i].c_str());
			continue;
			//return EXIT_FAILURE;
		}
		testImg.copyTo(img2);

		//load descriptors2
		//surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
		sift(img2.getMat(cv::ACCESS_READ), cv::Mat(), keypoints2, descriptors2);

		//drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );


		matcher.knnMatch(descriptors1, descriptors2, matches, 2);// Find two nearest matches
		matcher.match(descriptors2, descriptors1, backward_matches);

		std::vector<cv::DMatch> selected_matches;

		int cols = img2.cols;
		int rows = img2.rows;
		currM = findGoodMatches(cols, rows, keypoints1, keypoints2, matches, backward_matches, selected_matches,filenames[i].c_str());

		std::vector<cv::DMatch> good_matches;
		for( int i = 0; i < matches.size(); i++ )
		{
			if (matches[i][0].distance < ratio_Lowe * matches[i][1].distance)
			{
				cv::DMatch forward = matches[i][0];
				cv::DMatch backward = backward_matches[forward.trainIdx];
				if(backward.trainIdx == forward.queryIdx)
				{
					good_matches.push_back(forward);
				}

			}
			//good_matches.push_back( matches[i][0] );
			//good_matches.push_back( matches[i] );
		}


		cv::UMat colorImg;
		imread(filenames[i], CV_LOAD_IMAGE_COLOR).copyTo(colorImg);//get corresponding image

		cv::Mat img_matches;
		img_matches = drawGoodMatches(keypoints1, keypoints2, queryColorImg.getMat(cv::ACCESS_READ), colorImg.getMat(cv::ACCESS_READ), good_matches);
		cv::String outFile = output_location + "/" + RemoveFileExtension(SplitFilename(filenames[i])) + dataset_type;
		imwrite(outFile, img_matches);
		//allMatrices.push_back(currM);
		//std::cout<< "Matches Found:" <<good_matches.size() << std::endl<< std::endl;
		//matrices << filenames[i].c_str() << allMatrices.back();


		final_matches.push_back(selected_matches);
		matchesFound = selected_matches.size();


		if (matchesFound > maxMatches)
		{
			maxMatches = matchesFound;
			bestFileName = filenames[i];
			H = currM;
			bestKeypointsNum = keypoints2.size(); //This is the number of keypoints found on our best result image.
			bestMatches = good_matches;
			bestImageCenterX = img2.cols/2;
			bestImageCenterY = img2.rows/2;
		}

		matches.clear();
		backward_matches.clear();
		selected_matches.clear();
		keypoints2.clear();
		descriptors2.release();
	}


	//descriports and matrices are not needed anymore after this point
	//matrices.release();
	//f_verification.release();
	//e_verification.release();

	double currFitnessScore = 0.0;
	currFitnessScore = (double)maxMatches/ (double)bestKeypointsNum;
	std::cout <<"Image:\t"<<bestFileName<< "\nScore:\t"<< currFitnessScore << std::endl;

	//cv::Mat H = cv::Mat(currM, cv::Rect(0,0,3,currM.rows));
	//cv::Mat F = cv::Mat(currM, cv::Rect(3,0,3,currM.rows));
	//cv::Mat E = cv::Mat(currM, cv::Rect(6,0,3,currM.rows));
	//cv::Mat R = cv::Mat(currM, cv::Rect(9,0,6,currM.rows));
	//cv::Mat T = cv::Mat(currM, cv::Rect(15,0,2,currM.rows));

	//compute transform matrix then print
	std::vector<cv::Mat> oRvecs, oTvecs, oNvecs;

	cv::Mat CamMatrix = cv::Mat::eye(3, 3, CV_32F);

	//TODO: Set these from the params.yaml
	CamMatrix.at< float >(0, 0) = 500;
	CamMatrix.at< float >(1, 1) = 500;
	CamMatrix.at< float >(0, 2) = bestImageCenterX;
	CamMatrix.at< float >(1, 2) = bestImageCenterY;

	decomposeHomographyMat(H, CamMatrix, oRvecs, oTvecs, oNvecs);

	std::cout <<"Rotation matrices acquired:" << std::endl;
	double diff = 1;
	int selectedR = -1;//this is to force an error if no one was chosen
	for (int  j = 0; j < oRvecs.size(); ++j)
	{
			std::cout <<oRvecs[j]<< std::endl;
			//TODO: Remove the next line after testing
			std::cout <<oRvecs[j].at<double>(2,2)<< std::endl;
			double newDiff = 1 - oRvecs[j].at<double>(2,2);
			if (newDiff < diff)
			{
				diff = newDiff;
				selectedR = j;
			}
	}

	std::cout <<"Rotation matrix chosen:" << std::endl;
	std::cout <<oRvecs[selectedR]<< std::endl;


	std::cout <<"Orienting position..." << std::endl;

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

		/*
		for (int  j = 0; j < oRvecs.size(); ++j)
		{
			//std::cout<<j + 1<<std::endl;
			//std::cout<<oRvecs[j]<<std::endl<<std::endl;
			//the following lines are not guaranteed to be correct yet. Maybe an actual conversion is needed here.
			Eigen::Matrix3f m;
			cv2eigen(oRvecs[j], m);
			Eigen::Quaternionf q(m);
			cv::String tfStr = "rvec_frame" +std::to_string(j+1);
			tf::StampedTransform transform;
			tf::Quaternion t;
			tf::quaternionEigenToTF(Eigen::Quaterniond(q), t);
			//tf::()
			transform.setRotation(t);
			br[j].sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/kinect2_ir_optical_frame", tfStr));
		}
		*/
		r.sleep();
	}

	//writing to standard output part
	/*std::cout << bestFileName << " -> "<< currFitnessScore << std::endl
			<< "H = " << std::endl
			<<  H << std::endl << std::endl
			<< "F = " << std::endl
			<<  F << std::endl << std::endl
			<< "E = " << std::endl
			<<  E << std::endl << std::endl
			<< "R = " << std::endl
			<<  R << std::endl << std::endl
			<< "T = " << std::endl
			<<  T << std::endl << std::endl;
	 */
	//write image to disk
	//cv::Mat img_matches = drawGoodMatches(keypoints1, keypoints2, queryColorImg.getMat(cv::ACCESS_READ), bestImg.getMat(cv::ACCESS_READ), bestMatches);
	//while(img_matches.empty()){};

	//currFitnessScore *= 100;
	//cv::String fitnessValue = "Fitness: " + std::to_string(currFitnessScore) + '%';
	//putText(img_matches, fitnessValue, cv::Point(5, img1.rows - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
	//cv::String output_file = cwd + "output/" + bestFileName + dataset_type;
	//imwrite(output_file, img_matches);


	return EXIT_SUCCESS;
}
