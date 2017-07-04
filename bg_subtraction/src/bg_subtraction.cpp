#include <ros/ros.h>
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
#include <boost/foreach.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
//#include <signal.h>

typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloud;
tf::TransformListener * listener;
tf::TransformBroadcaster * br;
ros::Publisher pub;
image_transport::Publisher imagePub;
int fileCounter = 0;

//const std::string fileExt = ".pcd";
const std::string fileExt = ".jpg";

void callback(const PointCloud::ConstPtr& msg)
{
  
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  cloud->width    = msg->width;
  cloud->height   = msg->height;
  cloud->is_dense = false;
  cloud->points.resize (cloud->width * cloud->height);

  tf::StampedTransform transform;
  try{
    listener->lookupTransform("/camera_depth_optical_frame", "/iiwa_flange_link", ros::Time(0), transform);
    
    Eigen::Affine3d tranMat;
    tf::transformTFToEigen (transform, tranMat);
    Eigen::Matrix3d rotMat = tranMat.linear();
    Eigen::Vector3d zDirection = rotMat.col(2);

    //std::cout<< zDirection.transpose()<< std::endl;
    br->sendTransform(tf::StampedTransform(transform, ros::Time::now(), "/camera_depth_optical_frame", "iiwa_end_effector_link"));
    printf ("Cloud: width = %d, height = %d\n", msg->width, msg->height);

    int i = 0;

    BOOST_FOREACH (const pcl::PointXYZRGB& pt, msg->points)
    {
      //printf ("\t(%f, %f, %f)\n", pt.x, pt.y, pt.z);
      Eigen::Vector3d newVec(pt.x - transform.getOrigin().x(), pt.y - transform.getOrigin().y(), pt.z - transform.getOrigin().z());
      if (newVec.dot(zDirection) > 0)
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
    cv::Mat image(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::Vec3b intensity;
    //std::cout<<cloud_projected->width<<std::endl;
    //std::cout<<cloud_projected->height<<std::endl;
    i = 0;
    for(int v = 0; v < 480; ++v)
    {
      for(int u = 0; u < 640; ++u)
      {
        intensity.val[0] = cloud_projected->points[i].b;
        intensity.val[1] = cloud_projected->points[i].g;
        intensity.val[2] = cloud_projected->points[i].r;
        image.at<cv::Vec3b>(cv::Point(u,v)) = intensity;
        i++;
      }
    }

    cv_bridge::CvImage imageMsg;
    imageMsg.header.frame_id = msg->header.frame_id;
    imageMsg.header.stamp = ros::Time::now();
    imageMsg.encoding = "bgr8"; // Or whatever
    imageMsg.image    = image; // Your cv::Mat
    //sensor_msgs::ImagePtr imageMsg = cv_bridge::CvImage(, , image).toImageMsg();
    imagePub.publish(imageMsg.toImageMsg());
    
    //cloud->header.frame_id = msg->header.frame_id;
    //cloud->header.stamp = msg->header.stamp;

    //pub.publish(*cloud);
    
    std::ostringstream oss;
    oss << fileCounter << fileExt;
    std::cout << oss.str()<< " created." << std::endl;
    

    cv::imwrite(oss.str(), image);

    //pcl::io::savePCDFileASCII (oss.str(), *cloud);
    fileCounter++;
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
  ros::Subscriber sub = nh.subscribe<PointCloud>("/camera/depth_registered/points", 1, callback);
  pub = n.advertise<PointCloud>("/bg_subtracted", 1);
  image_transport::ImageTransport it(ng);
  imagePub = it.advertise("/bg_subtracted_image", 1);
   ros::spin();
}
