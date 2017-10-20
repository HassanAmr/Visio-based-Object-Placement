#include <ros/ros.h>
#include <rosbag/bag.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf_conversions/tf_eigen.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/surface/convex_hull.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <boost/foreach.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
//#include <signal.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
typedef message_filters::sync_policies::ApproximateTime<PointCloud, sensor_msgs::Image> SyncPolicy;

tf::TransformListener * listener;
tf::TransformBroadcaster * br;
ros::Publisher pub;
image_transport::Publisher imagePub;
rosbag::Bag bag;

int fileCounter = 0;
int validPixelCounter = 0;
int lastValidPixelCount = 0;

//const std::string fileExt = ".pcd";
const std::string fileExt = ".jpg";

void callback(const PointCloud::ConstPtr& msg, const sensor_msgs::Image::ConstPtr& imageRgb)
{
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->width    = msg->width;
  cloud->height   = msg->height;
  cloud->is_dense = false;
  cloud->points.resize (cloud->width * cloud->height);

  tf::StampedTransform transform;
  try{
    listener->lookupTransform("/kinect2_ir_optical_frame", "iiwa/sdh2_palm_link", ros::Time(0), transform);
    
    Eigen::Affine3d tranMat;
    tf::transformTFToEigen (transform, tranMat);
    Eigen::Matrix3d rotMat = tranMat.linear();
    Eigen::Vector3d zDirection = rotMat.col(2);

    //std::cout<< zDirection.transpose()<< std::endl;
    br->sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/kinect2_ir_optical_frame", "iiwa_end_effector_link"));
    //printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);

    int i = 0;

    BOOST_FOREACH (const pcl::PointXYZRGB& pt, msg->points)
    {
      //printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
      Eigen::Vector3d newVec(pt.x - transform.getOrigin().x(), pt.y - transform.getOrigin().y(), pt.z - transform.getOrigin().z());
      if (newVec.dot(zDirection) > -0.01)
      {
        cloud->points[i].x = pt.x;
        cloud->points[i].y = pt.y;
        cloud->points[i].z = pt.z;
        
        cloud->points[i].r = pt.r;
        cloud->points[i].g = pt.g;
        cloud->points[i].b = pt.b;
      }
      i++;
    }
    
    // Create a set of planar coefficients with X=Y=0,Z=1

    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    coefficients->values.resize (4);
    coefficients->values[0] = zDirection(0);
    coefficients->values[1] = zDirection(1);
    coefficients->values[2] = zDirection(2);
    coefficients->values[3] = ((-1)*(zDirection(0)*transform.getOrigin().x())) - (zDirection(1)*transform.getOrigin().y()) - (zDirection(2)*transform.getOrigin().z());//d= -a*x0 -b*y0 -c*z0

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_projected (new pcl::PointCloud<pcl::PointXYZRGB>);
    //cloud_projected->width    = cloud->width;
    //cloud_projected->height   = cloud->height;
    //cloud_projected->is_dense = false;
    //cloud_projected->points.resize (cloud->width * cloud->height);

    // Create the filtering object
    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (*cloud_projected);


    cloud_projected->header.frame_id = msg->header.frame_id;
    cloud_projected->header.stamp = msg->header.stamp;

    pub.publish(*cloud_projected);

    //cv_bridge::CvImagePtr

    cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat image_rgb(1080, 1920, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Mat image_rgb2(1080, 1920, CV_8UC3, cv::Scalar(255, 255, 255));



    cv::Vec3b intensity, rgb_intensity, rgb_intensity2;
    //std::cout<<cloud_projected->width<<std::endl;
    //std::cout<<cloud_projected->height<<std::endl;
    i = 0;
    int j = 0;
    //the following are used to hold the indexes of the first and last occurance of rgb within our pointcloud
    //so that the subsequent inner loop can get fill only the needed rgb values in our rgb image.
    int start_rgb, end_rgb, min_start = 1920, max_end = 0;
    validPixelCounter = 0;
    int first_v = 0, last_v;
    for(int v = 0; v < 1080; ++v)
    {
    	for(int u = 0; u < 1920; ++u)
    	{
    		intensity.val[0] = cloud_projected->points[i].b;
    		intensity.val[1] = cloud_projected->points[i].g;
    		intensity.val[2] = cloud_projected->points[i++].r;
    		image.at<cv::Vec3b>(cv::Point(u,v)) = intensity;

    		if(start_rgb == 0)
    		{
    			if(intensity != cv::Vec3b(0,0,0))
    			{
    				start_rgb = u;
    				if (start_rgb < min_start)
    				{
    					min_start = start_rgb;
    				}
    				if(first_v == 0)
    				{
    					first_v = v;
    				}
    			}
    		}
    		else
    		{
    			if(intensity != cv::Vec3b(0,0,0))
    			{
    				end_rgb = u;
    				last_v = v;
    				validPixelCounter++;
    			}
    		}
    		if (end_rgb > max_end )
    		{
    			max_end = end_rgb;
    		}
    	}

    	for (int u = 0; u < 1920; ++u)
    	{
    		if (u >= start_rgb && u < end_rgb)
    		{
    			rgb_intensity.val[0] = imageRgb->data[3*j];
    			rgb_intensity.val[1] = imageRgb->data[3*j + 1];
    			rgb_intensity.val[2] = imageRgb->data[3*j + 2];
    			image_rgb.at<cv::Vec3b>(cv::Point(u,v)) = rgb_intensity;
    		}
    		j++;
    	}


    	start_rgb = 0;
    	end_rgb = 0;
    }
    min_start-= 25;
    max_end += 	25;
    first_v -= 	25;
    last_v +=	25;
    j = 0;
    for(int v = 0; v < 1080; ++v)
    {
    	for (int u = 0; u < 1920; ++u)
    	{
    		if(u >= min_start && u < max_end && v >= first_v && v < last_v)
    		{
    			rgb_intensity.val[0] = imageRgb->data[3*j];
    			rgb_intensity.val[1] = imageRgb->data[3*j + 1];
    			rgb_intensity.val[2] = imageRgb->data[3*j + 2];
    			image_rgb2.at<cv::Vec3b>(cv::Point(u,v)) = rgb_intensity;
    		}
    		j++;
    	}
    }

    //std::cout<<min_start<<std::endl;
    //std::cout<<max_end<<std::endl;
    //std::cout<<first_v<<std::endl;
    //std::cout<<last_v<<std::endl;
    //std::cout<<min_start - 25<<std::endl;
    //std::cout<<first_v - 25<<std::endl;
    //std::cout<<max_end - min_start + 50<<std::endl;
    //std::cout<<last_v - first_v + 50<<std::endl;
    //cv_ptr->image.copyTo(image_rgb2(cv::Rect(min_start - 25, first_v - 25, max_end - min_start + 50, last_v - first_v + 50)));

    cv_bridge::CvImage imageMsg;
    imageMsg.header.frame_id = msg->header.frame_id;
    imageMsg.header.stamp = ros::Time::now();
    imageMsg.encoding = "bgr8"; // Or whatever
    imageMsg.image    = image_rgb2; // Your cv::Mat
    //sensor_msgs::ImagePtr imageMsg = cv_bridge::CvImage(, , image).toImageMsg();
    imagePub.publish(imageMsg.toImageMsg());

    //cloud->header.frame_id = msg->header.frame_id;
    //cloud->header.stamp = msg->header.stamp;

    //pub.publish(*cloud);
    
    //std::ostringstream oss;
    //oss << fileCounter << fileExt;
    //std::cout << oss.str()<< " created." << std::endl;
    

    //cv::imwrite(oss.str(), image);

    //std::ostringstream oss_rgb;
    //oss_rgb << fileCounter << "_rgb" << fileExt;
    //std::cout << oss_rgb.str()<< " created." << std::endl;

    if (validPixelCounter > lastValidPixelCount)
    {
    	cv::imwrite("bg_subtracted_image.jpg", image_rgb);
    	//cv::imwrite("bg_subtracted_image2.jpg", image_rgb2);
    	lastValidPixelCount = validPixelCounter;
    	//fileCounter++;
    }
    if (fileCounter == 0)
    {
    	//cv::imwrite("bg_subtracted_image.jpg", image_rgb);
    	cv::imwrite("bg_subtracted_image2.jpg", image_rgb2);
    	//lastValidPixelCount = validPixelCounter;
    	fileCounter++;
    }


    bag.write("point_cloud", ros::Time::now(), msg);
    bag.write("rgb_image", ros::Time::now(), imageRgb);


    //pcl::io::savePCDFileASCII (oss.str(), *cloud);
    //fileCounter++;
    //ros::shutdown();
  }
  catch (tf::TransformException &ex) {
    ROS_ERROR("%s",ex.what());
    ros::Duration(1.0).sleep();
  }
}
/*
void mySigintHandler(int sig)
{
  // Do some custom action.
  // For example, publish a stop message to some other nodes.
	printf ("Closing node...\n");

  // All the default sigint handler does is call shutdown()
  ros::shutdown();
}*/
int main(int argc, char** argv)
{
	ros::init(argc, argv, "bg_subtraction");
	ros::NodeHandle nh, ng, n;
	listener = new tf::TransformListener;
	br = new tf::TransformBroadcaster;
	bag.open("test.bag", rosbag::bagmode::Write);
	//ros::Subscriber sub = nh.subscribe<PointCloud>("/kinect2/hd/points", 1, callback);

	message_filters::Subscriber<PointCloud> registeredSub(nh, "/kinect2/hd/points", 1);
	message_filters::Subscriber<sensor_msgs::Image> rgbSub(nh, "/kinect2/hd/image_color", 1);
	message_filters::Synchronizer<SyncPolicy> sync(SyncPolicy(10), registeredSub, rgbSub);
	sync.registerCallback(boost::bind(&callback, _1, _2));

	pub = n.advertise<PointCloud>("/bg_subtracted", 1);
	image_transport::ImageTransport it(ng);
	imagePub = it.advertise("/bg_subtracted_image", 1);
	ros::spin();
}
