// src/image_publisher.cpp

#include "ros/ros.h"
#include "dv_ros_msgs/Event.h"
#include "dv_ros_msgs/EventArray.h"
#include "std_msgs/UInt8MultiArray.h"

class ImagePublisher {
public:
    ImagePublisher() : nh_("~"), image_data_(IMAGE_WIDTH * IMAGE_HEIGHT, 128) {
        ros::NodeHandle nh;

        // Subscribe to the /events topic that provides dv_ros_msgs::EventArray
        sub_ = nh.subscribe("/events", 1, &ImagePublisher::eventArrayCallback, this);

        // Publish a UInt8MultiArray to /output/event
        image_pub_ = nh_.advertise<std_msgs::UInt8MultiArray>("/output/event", 1);

        // Publish at 30 Hz
        timer_ = nh_.createTimer(ros::Duration(1.0 / PUBLISH_RATE), 
                                 &ImagePublisher::timerCallback, this);
    }

    void eventArrayCallback(const dv_ros_msgs::EventArray::ConstPtr& msg) {
        // Debug info
        ROS_INFO("Current ROS Time: %f", ros::Time::now().toSec());
        ROS_INFO("Header Time of Received Message: %f", msg->header.stamp.toSec());

        // Reset the image_data_ buffer to a neutral gray (128)
        std::fill(image_data_.begin(), image_data_.end(), 128);

        // Traverse each event and update our pixel intensities accordingly
        for (const auto& event : msg->events) {
            if (event.x < IMAGE_WIDTH && event.y < IMAGE_HEIGHT) {
                // Check polarity explicitly: +1 for positive, -1 for negative
                if (event.polarity) {
                    // Positive event => increment
                    if (image_data_[event.y * IMAGE_WIDTH + event.x] < 255) {
                        image_data_[event.y * IMAGE_WIDTH + event.x]++;
                    }
                } 
                else{
                    // Negative event => decrement
                    if (image_data_[event.y * IMAGE_WIDTH + event.x] > 0) {
                        image_data_[event.y * IMAGE_WIDTH + event.x]--;
                    }
                }
            }
        }
    }

    void timerCallback(const ros::TimerEvent&) {
        // Construct a UInt8MultiArray to publish our image data
        std_msgs::UInt8MultiArray array_msg;
        array_msg.layout.dim.push_back(std_msgs::MultiArrayDimension());
        array_msg.layout.dim[0].label = "data";
        array_msg.layout.dim[0].size = image_data_.size();
        array_msg.layout.dim[0].stride = image_data_.size();
        array_msg.layout.data_offset = 0;

        array_msg.data = image_data_;  // Transfer pixel data
        image_pub_.publish(array_msg);
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber sub_;
    ros::Publisher image_pub_;
    ros::Timer timer_;
    std::vector<uint8_t> image_data_;

    static const int IMAGE_WIDTH = 640;
    static const int IMAGE_HEIGHT = 480;
    static const int PUBLISH_RATE = 30;
};

int main(int argc, char **argv) {
    ros::init(argc, argv, "image_creator_node");

    ImagePublisher image_publisher;

    ros::spin();
    return 0;
}
