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


float ratio_Lowe = 0.7f; // As in Lowe's paper; can be tuned
const cv::String dataset_type = ".jpg";
cv::String query_location, test_location, cwd, output_location, log_location;//cwd is short for curring working directory


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

void drawBadMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& _img1,
    const cv::Mat& _img2,
    std::vector< std::vector<cv::DMatch> >& bad_matches,
    std::vector<cv::DMatch>& backward_matches,
    std::string currImgText
    )
{
	std::cout<<"Bad Match!"<<std::endl;
	std::vector< cv::DMatch > good_matches;
	for( int i = 0; i < bad_matches.size(); i++ )
	{
		if (bad_matches[i][0].distance < ratio_Lowe * bad_matches[i][1].distance)
		{
			cv::DMatch forward = bad_matches[i][0];
			cv::DMatch backward = backward_matches[forward.trainIdx];
			if(backward.trainIdx == forward.queryIdx)
			{
				good_matches.push_back(forward);
			}
		}
	}
	//-- Localize the object
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;

	for( size_t i = 0; i < good_matches.size(); i++ )
	{
		//-- Get the keypoints from the good matches
		obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
		scene.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
	}

	cv::String outFile = output_location + "/Bad_" + RemoveFileExtension(SplitFilename(currImgText));

	// drawing the results

	cv::Size img1size = _img1.size(), img2size = _img2.size();
	cv::Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height) );
	cv::Mat canvas = cv::Mat(size, CV_8UC3, cv::Scalar(255,255,255));


	std::vector<cv::Point> pts;
	for (size_t i = 0; i < keypoints1.size(); i++)
	{
		pts.push_back(keypoints1[i].pt);
	}
	std::vector<cv::Point> hull;
	cv::convexHull(pts,hull);

	cv::Mat mask(img1size, CV_8UC3, cv::Scalar(255,255,255));
	cv::fillConvexPoly(mask, hull, cv::Scalar(0,0,0));

	cv::Mat img1 = cv::Mat(img1size, CV_8UC3, cv::Scalar(255,255,255));
	cv::Mat img2;

	bitwise_or(_img1, mask, img1);
	//_img1.copyTo(img1);
	_img2.copyTo(img2);

	//start drawings
	cv::Mat img_keypoints = canvas;
	cv::Mat img1_keypoints, img2_keypoints;
	cv::drawKeypoints(img1,keypoints1,img1_keypoints);
	cv::drawKeypoints(img2,keypoints2,img2_keypoints);

	img1_keypoints.copyTo(img_keypoints(cv::Rect(0, 0, img1_keypoints.cols, img1_keypoints.rows)));
	img2_keypoints.copyTo(img_keypoints(cv::Rect(img1_keypoints.cols, 0, img2_keypoints.cols, img2_keypoints.rows)));
	imwrite(outFile + "_keypoints" + dataset_type, img_keypoints);


	img1.copyTo(canvas(cv::Rect(0, 0, img1.cols, img1.rows)));
	img2.copyTo(canvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

	cv::Mat img_matches;
	canvas.copyTo(img_matches);

	drawMatches( img1, keypoints1, img2, keypoints2,
			good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
			std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
	imwrite(outFile + "_matches" + dataset_type, img_matches);




	cv::Mat img_rotation;
	canvas.copyTo(img_rotation);

	std::vector<cv::Point2f> obj_corners(hull.size());
	std::vector<cv::Point2f> scene_corners(hull.size());

	for(int i = 0; i < hull.size();i++)
	{
		obj_corners[i] = hull[i];
	}

	//-- Draw lines between the corners (the mapped object in the scene - image_2 )
	for(int i = 0; i < obj_corners.size();i++)
	{
		line( img_rotation,
				obj_corners[i], obj_corners[(i+1)%obj_corners.size()],
				cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );

	}

	if (good_matches.size() > 0)
	{
		cv::Mat H = findHomography( obj, scene, cv::RANSAC, 3 );


		if (countNonZero(H) > 0 )
		{
			cv::perspectiveTransform( obj_corners, scene_corners, H);
			cv::Point2f offset = cv::Point2f( (float)img1.cols, 0);
			for(int i = 0; i < scene_corners.size();i++)
			{
				line( img_rotation,
						scene_corners[i] + offset, scene_corners[(i+1)%scene_corners.size()] + offset,
						cv::Scalar( 0, 255, 0), 2, cv::LINE_AA );

			}

			int ptIndex = 0;
			//cv::RNG rng(12345);
			int color = 0;
			for (int i = 0; i < 4;i++)
			{
				ptIndex = (((float)i/(float)4)*obj_corners.size());
				cv::Point p1 = obj_corners[ptIndex];
				cv::Point p2 = scene_corners[ptIndex] + offset;
				cv::LineIterator it(img_rotation, p1, p2, 8);            // get a line iterator
				//color = rng.uniform(0,255);
				if ((i % 2) == 0){
					//color = 255;
					line( img_rotation, p1, p2, cv::Scalar( 0, 0, 255), 1, cv::LINE_AA );
				}
				else{
					//color = 0;
					line( img_rotation, p1, p2, cv::Scalar( 0, 0, 0), 1, cv::LINE_AA );

				}
				//for(int j = 0; j < it.count; j++,it++)
				//    if ( j%5!=0 ) {(*it)[2] = color;}         // every 5'th pixel gets dropped, red or black stipple line
			}
		}
	}
	imwrite(outFile + "_rotation" + dataset_type, img_rotation);
}

void drawGoodMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    const cv::Mat& _img1,
    const cv::Mat& _img2,
	const cv::Mat H,
    std::vector<cv::DMatch>& good_matches,
    std::string currImgText
    )
{
    cv::String outFile = output_location + "/" + RemoveFileExtension(SplitFilename(currImgText));

    // drawing the results

    cv::Size img1size = _img1.size(), img2size = _img2.size();
    cv::Size size( img1size.width + img2size.width, MAX(img1size.height, img2size.height) );
	cv::Mat canvas = cv::Mat(size, CV_8UC3, cv::Scalar(255,255,255));


    std::vector<cv::Point> pts;
    for (size_t i = 0; i < keypoints1.size(); i++)
    {
    	pts.push_back(keypoints1[i].pt);
    }
    std::vector<cv::Point> hull;
    cv::convexHull(pts,hull);

    cv::Mat mask(img1size, CV_8UC3, cv::Scalar(255,255,255));
    cv::fillConvexPoly(mask, hull, cv::Scalar(0,0,0));

    cv::Mat img1 = cv::Mat(img1size, CV_8UC3, cv::Scalar(255,255,255));
    cv::Mat img2;

    bitwise_or(_img1, mask, img1);
    //_img1.copyTo(img1);
    _img2.copyTo(img2);

    //start drawings
	cv::Mat img_keypoints = canvas;
    cv::Mat img1_keypoints, img2_keypoints;
    cv::drawKeypoints(img1,keypoints1,img1_keypoints);
    cv::drawKeypoints(img2,keypoints2,img2_keypoints);

	img1_keypoints.copyTo(img_keypoints(cv::Rect(0, 0, img1_keypoints.cols, img1_keypoints.rows)));
	img2_keypoints.copyTo(img_keypoints(cv::Rect(img1_keypoints.cols, 0, img2_keypoints.cols, img2_keypoints.rows)));
    imwrite(outFile + "_keypoints" + dataset_type, img_keypoints);


   	img1.copyTo(canvas(cv::Rect(0, 0, img1.cols, img1.rows)));
    img2.copyTo(canvas(cv::Rect(img1.cols, 0, img2.cols, img2.rows)));

    cv::Mat img_matches;
    canvas.copyTo(img_matches);

    drawMatches( img1, keypoints1, img2, keypoints2,
                 good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
                 std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS | cv::DrawMatchesFlags::DRAW_OVER_OUTIMG );
    imwrite(outFile + "_matches" + dataset_type, img_matches);




    cv::Mat img_rotation;
    canvas.copyTo(img_rotation);

    std::vector<cv::Point2f> obj_corners(hull.size());
    std::vector<cv::Point2f> scene_corners(hull.size());

    for(int i = 0; i < hull.size();i++)
    {
    	obj_corners[i] = hull[i];
    }
    //cv::Mat H = findHomography( obj, scene, cv::RANSAC );
    cv::perspectiveTransform( obj_corners, scene_corners, H);

    //-- Draw lines between the corners (the mapped object in the scene - image_2 )
	for(int i = 0; i < obj_corners.size();i++)
	{
		line( img_rotation,
				obj_corners[i], obj_corners[(i+1)%obj_corners.size()],
				cv::Scalar( 255, 0, 0), 2, cv::LINE_AA );

	}


	cv::Point2f offset = cv::Point2f( (float)img1.cols, 0);
	for(int i = 0; i < scene_corners.size();i++)
	{
		line( img_rotation,
		          scene_corners[i] + offset, scene_corners[(i+1)%scene_corners.size()] + offset,
				  cv::Scalar( 255, 0, 0), 2, cv::LINE_AA );

	}

	int ptIndex = 0;
	//cv::RNG rng(12345);
	int color = 0;
	for (int i = 0; i < 4;i++)
	{
		ptIndex = (((float)i/(float)4)*obj_corners.size());
		cv::Point p1 = obj_corners[ptIndex];
		cv::Point p2 = scene_corners[ptIndex] + offset;
		cv::LineIterator it(img_rotation, p1, p2, 8);            // get a line iterator
		//color = rng.uniform(0,255);
		if ((i % 2) == 0){
			//color = 255;
			line( img_rotation, p1, p2, cv::Scalar( 0, 0, 255), 1, cv::LINE_AA );
		}
		else{
			//color = 0;
			line( img_rotation, p1, p2, cv::Scalar( 0, 0, 0), 1, cv::LINE_AA );

		}
		//for(int j = 0; j < it.count; j++,it++)
		//    if ( j%5!=0 ) {(*it)[2] = color;}         // every 5'th pixel gets dropped, red or black stipple line
	}
    imwrite(outFile + "_rotation" + dataset_type, img_rotation);
}

//later should be void again, and Mat should go to the new function that will be specifically responsible for homography
cv::Mat findGoodMatches_debug(
    int cols, int rows,//the columns and rows that cover exactly how big the object is, used for RANSAC homography
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    std::vector< std::vector<cv::DMatch> >& matches,
    std::vector<cv::DMatch>& backward_matches,
    std::vector<cv::DMatch>& selected_matches,
    std::string currImgText
    )
{
	std::cout<<currImgText<<std::endl;
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
    int last_good_matches = 0;
	cv::Mat H;//, M; //The matrices to be returned
	std::vector<cv::KeyPoint>  newKeypoints2;
	for (int i=0; i<keypoints2.size(); i++)
		newKeypoints2.push_back(keypoints2[i]);
	//new_matches = good_matches;
	int drawCounter = 0;
    while (1)
	{
    //-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for( size_t i = 0; i < good_matches.size(); i++ )
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
			scene.push_back( newKeypoints2[ good_matches[i].trainIdx ].pt );
		}

		if (obj.size() > 0)
		{
			std::vector<cv::Point> pts;
			for (size_t i = 0; i < keypoints1.size(); i++)
			{
				pts.push_back(keypoints1[i].pt);
			}
			std::vector<cv::Point> hull;
			cv::convexHull(pts,hull);
			std::vector<cv::Point2f> obj_corners(hull.size());
			std::vector<cv::Point2f> scene_corners(hull.size());

			for(int i = 0; i < hull.size();i++)
			{
				obj_corners[i] = hull[i];
			}

			H = findHomography( obj, scene, cv::RANSAC, 3 );

			if (countNonZero(H) < 1)
			{
				break;
			}
			else
			{
				cv::perspectiveTransform( obj_corners, scene_corners, H);
				std::cout<<"scene_corners: "<< std::endl;
				std::cout<<scene_corners << std::endl;

				cv::Mat drawing = cv::Mat::zeros(rows,cols, CV_8UC1); //opencv deals with matrix

				for(int i = 0; i < scene_corners.size();i++)
				{
					line( drawing,
							scene_corners[i], scene_corners[(i+1)%scene_corners.size()],// + offset,
							cv::Scalar( 255 ), 3, 8 );
				}

				cv::String outFile = "/home/hassan/tmp/drawings/" + RemoveFileExtension(SplitFilename(currImgText)) + "_drawing_" + std::to_string(drawCounter) + ".jpg";
				imwrite(outFile, drawing);

				cv::UMat tempImg, tempImg2;
				cv::imread(currImgText, CV_LOAD_IMAGE_COLOR).copyTo(tempImg);//get corresponding image
				cv::imread(currImgText, CV_LOAD_IMAGE_COLOR).copyTo(tempImg2);//get corresponding image
				for(int i = 0; i < scene_corners.size();i++)
				{
					line( tempImg,
							scene_corners[i], scene_corners[(i+1)%scene_corners.size()],// + offset,
							cv::Scalar( 255 ), 3, 8 );
					line( tempImg2,
							scene_corners[i], scene_corners[(i+1)%scene_corners.size()],// + offset,
							cv::Scalar( 255 ), 3, 8 );
				}

				cv::String outFile1 = "/home/hassan/tmp/drawings/" + RemoveFileExtension(SplitFilename(currImgText)) + "_check1_" + std::to_string(drawCounter) + ".jpg";
				for ( size_t i = 0; i < good_matches.size(); i++ )
				{
					std::vector<cv::KeyPoint> newK;
					newK.push_back(keypoints2[ good_matches[i].trainIdx ]);
					cv::drawKeypoints(tempImg,newK, tempImg, cv::Scalar( 0, 255, 0));
				}
				cv::imwrite(outFile1, tempImg);
				std::vector<cv::KeyPoint> tempKeys;
				//find contours of the above drawn region
				double test = 0.0;
				//cv::UMat tempImg2;
				//cv::imread(currImgText, CV_LOAD_IMAGE_COLOR).copyTo(tempImg2);//get corresponding image
				cv::String outFile2 = "/home/hassan/tmp/drawings/" + RemoveFileExtension(SplitFilename(currImgText)) + "_check2_" + std::to_string(drawCounter++) + ".jpg";

			    //std::vector<std::vector<cv::Point> > contours;
			    //contours.push_back(hull);
				//cv::drawContours( tempImg2, contours, 0, cv::Scalar( 255 ), 2, 8);

				for( size_t i = 0; i < newKeypoints2.size(); i++ )
					std::cout<<i<<": " << newKeypoints2[i].pt<< std::endl;
				std::cout<< "-------------------------------------------------------------------------------"<< std::endl;

				for( size_t i = 0; i < good_matches.size(); i++ )
				{
					std::cout<<i<<": " <<keypoints2[ good_matches[i].trainIdx ].pt<< std::endl;

					test = pointPolygonTest( scene_corners, keypoints2[ good_matches[i].trainIdx ].pt , false );

					if (test >= 0)
					{
						selected_matches.push_back(good_matches[i]);
						tempKeys.push_back(keypoints2[ good_matches[i].trainIdx ]);
					}
				}

				std::cout<< "good_matches: " <<selected_matches.size()<< std::endl;

				cv::drawKeypoints(tempImg2,tempKeys, tempImg2, cv::Scalar( 0, 255, 0));
				cv::imwrite(outFile2, tempImg2);
				if (last_good_matches == selected_matches.size())
				{
					break;
				}
				else
				{
					last_good_matches = selected_matches.size();
					good_matches.clear();
					newKeypoints2.clear();
					for(int i = 0; i < selected_matches.size();i++)
						good_matches.push_back(selected_matches[i]);
					for(int i = 0; i < tempKeys.size();i++)
						newKeypoints2.push_back(tempKeys[i]);

					selected_matches.clear();
					tempKeys.clear();
				}
			}
		}
		else
		{
			break;
		}
	}

    return H;
}

cv::Mat findGoodMatches(
    const std::vector<cv::KeyPoint>& keypoints1,
    const std::vector<cv::KeyPoint>& keypoints2,
    std::vector< std::vector<cv::DMatch> >& matches,
    std::vector<cv::DMatch>& backward_matches,
    std::vector<cv::DMatch>& selected_matches
    )
{
    std::vector< cv::DMatch > good_matches;
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
    }
    int last_good_matches = 0;
	cv::Mat H;
	//because we need keypoints2 unaffected to draw them at the end.
	std::vector<cv::KeyPoint>  newKeypoints2;
		for (int i=0; i<keypoints2.size(); i++)
			newKeypoints2.push_back(keypoints2[i]);
    while (1)
	{
    //-- Localize the object
		std::vector<cv::Point2f> obj;
		std::vector<cv::Point2f> scene;

		for( size_t i = 0; i < good_matches.size(); i++ )
		{
			//-- Get the keypoints from the good matches
			obj.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
			scene.push_back( newKeypoints2[ good_matches[i].trainIdx ].pt );
		}

		if (obj.size() > 0)
		{
			std::vector<cv::Point> pts;
			for (size_t i = 0; i < keypoints1.size(); i++)
			{
				pts.push_back(keypoints1[i].pt);
			}
			std::vector<cv::Point> hull;
			cv::convexHull(pts,hull);
			std::vector<cv::Point2f> obj_corners(hull.size());
			std::vector<cv::Point2f> scene_corners(hull.size());

			for(int i = 0; i < hull.size();i++)
			{
				obj_corners[i] = hull[i];
			}

			H = findHomography( obj, scene, cv::RANSAC, 3 );
			if (countNonZero(H) < 1)
			{
				break;
			}
			else
			{
				cv::perspectiveTransform( obj_corners, scene_corners, H);

				std::vector<cv::KeyPoint> tempKeys;
				double test = 0.0;
				for( size_t i = 0; i < good_matches.size(); i++ )
				{
					test = pointPolygonTest( scene_corners, keypoints2[ good_matches[i].trainIdx ].pt , false );

					if (test >= 0)
					{
						selected_matches.push_back(good_matches[i]);
						tempKeys.push_back(keypoints2[ good_matches[i].trainIdx ]);
					}
				}
				if (last_good_matches == selected_matches.size())//check whether we converged or not
				{
					break;
				}
				else
				{
					last_good_matches = selected_matches.size();
					good_matches.clear();
					newKeypoints2.clear();
					for(int i = 0; i < selected_matches.size();i++)
						good_matches.push_back(selected_matches[i]);
					for(int i = 0; i < tempKeys.size();i++)
						newKeypoints2.push_back(tempKeys[i]);

					selected_matches.clear();
					tempKeys.clear();
				}
			}
		}
		else{
			break;
		}
	}
    return H;
}

int main(int argc, char **argv)
{

	ros::init(argc, argv, "object_transormations");

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

	//-------------------------------------------------------------------------------
	if (!(nh.getParam("/visiobased_placement/LOWE_RATIO", ratio_Lowe)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	bool debug_matches = false;
	if (!(nh.getParam("/visiobased_placement/DEBUG_MATCHING", debug_matches)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}

	int focal_lenth = 500;
	if (!(nh.getParam("/visiobased_placement/CAMERA_FL", focal_lenth)))
	{
		printf("Failure with input parameter.\nProgram will exit with failure status.");
		return EXIT_FAILURE;
	}



	//std::cout << focal_lenth << std::endl;
	//std::cout << query_location << std::endl;
	//std::cout << test_location << std::endl;
	//std::cout << output_location << std::endl;

	query_location = cwd + query_location;
	test_location = cwd + test_location;
	output_location = cwd + output_location;
	log_location = cwd + log_location;


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
	double currFitnessScore = 0.0;
	double maxFitnessScore = 0.0;


	cv::Mat H;

	std::ofstream logOutput;
	cv::String logName = log_location + "/image_rotation.csv";
	logOutput.open (logName.c_str(), std::ios::out | std::ios::app);
    logOutput << "File" <<"\t"<< "Matches" <<"\t" << "Keypoints" <<"\t" <<"Fitness"<< "\t"<< "Min Dist" <<"\t"<<"Max Dist"<<"\t"<< "Time"<<std::endl;


	int64 t0_total = cv::getTickCount();
	double time_lost = 0;//this will be the accumulation of time spent drawing, debugging, and writing to disk.
	if (filenames.size() < 1)
	{
		ROS_WARN("No files processed.\nWaiting for user to terminate.");
		return EXIT_SUCCESS;
		//while (1){}
	}
	for (size_t i=0; i< filenames.size(); i++)
	{

		ROS_INFO("%s", filenames[i].c_str());

		cv::UMat img2;
		cv::UMat testImg;

		int reduced = 0;
		try
		{
			imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE).copyTo(testImg);//get corresponding image

			if (MAX(testImg.rows, testImg.cols) > 3000)
			{
				testImg.release();
				imread(filenames[i], cv::IMREAD_REDUCED_GRAYSCALE_4).copyTo(testImg);//get corresponding image
				reduced = 4;

			}
			else if (MAX(testImg.rows, testImg.cols) > 1500)
			{
				testImg.release();
				imread(filenames[i], cv::IMREAD_REDUCED_GRAYSCALE_2).copyTo(testImg);//get corresponding image
				reduced = 2;
			}

		}
		catch (const std::exception& e)
		{
			ROS_WARN("Something went wrong when loading the test image %s. Skipping over it.", filenames[i].c_str());
			continue;
		}

		//testImg = imread(filenames[i], CV_LOAD_IMAGE_GRAYSCALE).getUMat( cv::ACCESS_READ );
		if(queryImg.empty())
		{
			std::cout << "Couldn't load " << test_location << std::endl;
			//cmd.printMessage();
			printf("Something went wrong loading %s.\nSkipping over it.", filenames[i].c_str());
			continue;
			//return EXIT_FAILURE;
		}
		testImg.copyTo(img2);

		//--------------------------------------------------------------------------------------------------------------------------------------
	    //Here we begin the procces of feature detection, matching, then optimization
		int64 t0 = cv::getTickCount();

		//load descriptors2
		//surf(img2.getMat(ACCESS_READ), Mat(), keypoints2, descriptors2);
		sift(img2.getMat(cv::ACCESS_READ), cv::Mat(), keypoints2, descriptors2);

		matcher.knnMatch(descriptors1, descriptors2, matches, 2);// Find two nearest matches
		matcher.match(descriptors2, descriptors1, backward_matches);

		std::vector<cv::DMatch> selected_matches;

	    //-- Sort matches
		//We are only allowed to sort matches. We can never touch backwards, since we use trainIdx (descriptors2 index) to fetch the relevant match from backwards matches.
	    std::sort(matches.begin(), matches.end());
	    double minDist = matches.front()[0].distance;
	    double maxDist = matches.back()[0].distance;

	    try
	    {
	    	currM = findGoodMatches(keypoints1, keypoints2, matches, backward_matches, selected_matches);
	    }
	    catch (const std::exception& e)
	    {
			ROS_WARN("Something is not right with the matching results for file:\t %s", filenames[i].c_str());
	    	continue;
	    }
	    int64 t1 = cv::getTickCount();
		//--------------------------------------------------------------------------------------------------------------------------------------
	    double secs = (t1-t0)/cv::getTickFrequency();


	    matchesFound = selected_matches.size();
		currFitnessScore = (double)matchesFound/ (double)keypoints2.size();
	    ROS_INFO("%d matches found in %f seconds with %f%% fitness", selected_matches.size(), secs, (currFitnessScore*100));
	    ROS_INFO("Min Dist:\t %f\t Max Dist:\t %f", minDist, maxDist);

	    logOutput << filenames[i].c_str() <<"\t"<< selected_matches.size()<< "\t" << keypoints2.size() << "\t" <<currFitnessScore << "\t" <<minDist<<"\t"<< maxDist <<"\t"<< secs <<std::endl;

	    //######################################################################################################################################
	    //Subtract the following from your total time.
		int64 t0_lost = cv::getTickCount();

	    if (debug_matches)
	    {
	    	//Run debug version and check if they are the same
			std::vector<cv::DMatch> debug_matches;
	    	int cols = img2.cols;
			int rows = img2.rows;
			cv::Mat debugM = findGoodMatches_debug(cols, rows, keypoints1, keypoints2, matches, backward_matches, debug_matches,filenames[i].c_str());
			cv::Mat temp;
			cv::bitwise_xor(currM,debugM,temp); //It vectorizes well with SSE/NEON

			if (cv::countNonZero(temp) || (debug_matches.size() != selected_matches.size()))
			{
				ROS_WARN("Something is not right with the matching results. Debug matches found: %d", debug_matches.size());
				std::cout<<"H:"<<std::endl<<currM<<std::endl;
				std::cout<<"Debug H:"<<std::endl<<debugM<<std::endl;
			}
	    }
		cv::UMat colorImg;

		try
		{
			if (reduced == 4)
			{
				imread(filenames[i], cv::IMREAD_REDUCED_COLOR_4).copyTo(colorImg);//get corresponding image

			}
			else if (reduced == 2)
			{
				imread(filenames[i], cv::IMREAD_REDUCED_COLOR_2).copyTo(colorImg);//get corresponding image
			}
			else
			{
				imread(filenames[i], CV_LOAD_IMAGE_COLOR).copyTo(colorImg);//get corresponding image
			}
		}
		catch (const std::exception& e)
		{
			ROS_WARN("Something went wrong when loading the image. File is probably too big. Will load greyscale image instead.");
			continue;
		}

		cv::Mat img_matches;
		try
		{
			if (selected_matches.size() > 0)
			{
				drawGoodMatches(keypoints1, keypoints2, queryColorImg.getMat(cv::ACCESS_READ), colorImg.getMat(cv::ACCESS_READ), currM, selected_matches, filenames[i].c_str());
			}
			else
			{
				drawBadMatches(keypoints1, keypoints2, queryColorImg.getMat(cv::ACCESS_READ), colorImg.getMat(cv::ACCESS_READ), matches, backward_matches, filenames[i].c_str());
			}
		}
		catch (const std::exception& e)
		{
			ROS_WARN("Something went wrong when drawing results for file:\t %s", filenames[i].c_str());
			continue;
		}
		int64 t1_lost = cv::getTickCount();
		time_lost += (t1_lost-t0_lost)/cv::getTickFrequency();
	    //######################################################################################################################################

		if (currFitnessScore > maxFitnessScore)
		{
			maxFitnessScore = currFitnessScore;
			bestFileName = filenames[i];
			H = currM;
			bestKeypointsNum = keypoints2.size(); //This is the number of keypoints found on our best result image.
			bestMatches = selected_matches;
			bestImageCenterX = img2.cols/2;
			bestImageCenterY = img2.rows/2;
		}

		matches.clear();
		backward_matches.clear();
		selected_matches.clear();
		keypoints2.clear();
		descriptors2.release();
	}
	if (maxFitnessScore == 0)
	{
		ROS_WARN("No Good results.\nWaiting for user to terminate.");
		return EXIT_SUCCESS;
		//while (1){}
	}
	//compute transform matrix then print
	std::vector<cv::Mat> oRvecs, oTvecs, oNvecs;

	cv::Mat CamMatrix = cv::Mat::eye(3, 3, CV_32F);

	CamMatrix.at< float >(0, 0) = focal_lenth;
	CamMatrix.at< float >(1, 1) = focal_lenth;
	CamMatrix.at< float >(0, 2) = bestImageCenterX;
	CamMatrix.at< float >(1, 2) = bestImageCenterY;

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
	//Stop the clock!
	int64 t1_total = cv::getTickCount();
    double secs = (t1_total-t0_total)/cv::getTickFrequency();
    secs -= time_lost;
    logOutput <<std::endl;

    logOutput << "Total" <<"\t"<< "-" <<"\t"<< "-" <<"\t"<< "-" <<"\t"<< "-" <<"\t"<< "-" <<"\t"<< secs <<std::endl;

	std::cout <<"Rotation matrix chosen:" << std::endl;
	std::cout <<oRvecs[selectedR]<< std::endl;


	std::cout <<"Orienting position..." << std::endl;
	std::cout <<"Image:\t"<<bestFileName<< "\nScore:\t"<< maxFitnessScore << std::endl;
	logOutput.close();
	std::ofstream rotOutput;
	cv::String rotOutName = log_location + "/rotation_matrix.csv";
	logOutput.open (rotOutName.c_str(), std::ios::out | std::ios::app);
    logOutput << bestFileName <<std::endl;
    logOutput << oRvecs[selectedR].at<double>(0, 0)<<"\t"<< oRvecs[selectedR].at<double>(0, 1)<<"\t" << oRvecs[selectedR].at<double>(0, 2) <<std::endl;
    logOutput << oRvecs[selectedR].at<double>(1, 0)<<"\t"<< oRvecs[selectedR].at<double>(1, 1)<<"\t" << oRvecs[selectedR].at<double>(1, 2) <<std::endl;
    logOutput << oRvecs[selectedR].at<double>(2, 0)<<"\t"<< oRvecs[selectedR].at<double>(2, 1)<<"\t" << oRvecs[selectedR].at<double>(2, 2) <<std::endl;
    logOutput.close();
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
