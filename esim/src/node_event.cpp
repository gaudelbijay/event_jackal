#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <dv_ros_msgs/Event.h>
#include <dv_ros_msgs/EventArray.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <mutex>
#include <vector>
#include <string>
#include <stdexcept>

// Include the C++ wrapper for your CUDA code.
#include "esim_cuda.cpp"

// For convenient tensor indexing.
using namespace torch::indexing;

// Helper struct to hold a frame and its timestamp.
struct FrameData {
  cv::Mat frame;
  ros::Time timestamp;
};

class EventCameraCudaNode {
public:
  EventCameraCudaNode() {
    ros::NodeHandle private_nh("~");

    // Load parameters.
    private_nh.param("ct_pos", ct_pos_, 0.2f);
    private_nh.param("ct_neg", ct_neg_, 0.2f);
    // No log transform: we do not load/use use_log and log_eps.
    private_nh.param("width", width_, 640);
    private_nh.param("height", height_, 480);
    int refrac_ns = 0;
    private_nh.param("refractory_period_ns", refrac_ns, 0);
    refractory_period_ns_ = static_cast<int64_t>(refrac_ns);

    double accumulation_period = 0.033;
    private_nh.param("accumulation_period", accumulation_period, 0.033);
    // Optional: maximum allowed buffer age (in seconds). If set > 0, frames older than this will be dropped.
    private_nh.param("max_buffer_age", max_buffer_age_, 0.0); // 0 means disabled.

    // Setup publishers and subscribers.
    event_pub_ = nh_.advertise<dv_ros_msgs::EventArray>("events", 1);
    image_sub_ = nh_.subscribe("/realsense/color/image_raw", 4,
                               &EventCameraCudaNode::imageCallback, this);
    collision_sub_ = nh_.subscribe("collision", 1,
                                   &EventCameraCudaNode::collisionCallback, this);

    timer_ = nh_.createTimer(ros::Duration(accumulation_period),
                             &EventCameraCudaNode::timerCallback, this);

    ROS_INFO("EventCameraCudaNode started with: ct_pos=%.3f, ct_neg=%.3f, refractory_ns=%ld, resolution=%dx%d",
             ct_pos_, ct_neg_, refractory_period_ns_, width_, height_);
  }

private:
  // Callback for incoming images.
  void imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    } catch (const cv_bridge::Exception& e) {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat preprocessed = preprocess(cv_ptr->image, msg->encoding);
    if (preprocessed.empty())
      return;
  
    std::lock_guard<std::mutex> lock(frame_vector_mutex_);
    // Limit the frame buffer to a maximum of 5 frames.
    if (frame_vector_.size() >= 5) {
      frame_vector_.erase(frame_vector_.begin());  // Remove the oldest frame.
    }
    FrameData fd;
    fd.frame = preprocessed;
    fd.timestamp = msg->header.stamp;
    frame_vector_.push_back(fd);
  }
  

  // Callback for collision messages.
  // When a collision occurs, reset the entire frame buffer and internal state.
  void collisionCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data) {
      ROS_WARN("Collision detected! Resetting event simulator buffer.");
      resetBuffer();
    }
  }

  // Optionally check buffer age and clear if too old.
  void checkBufferAge() {
    if (max_buffer_age_ > 0.0 && !frame_vector_.empty()) {
      ros::Duration age = ros::Time::now() - frame_vector_.front().timestamp;
      if (age.toSec() > max_buffer_age_) {
        ROS_WARN("Frame buffer age (%.2f s) exceeds maximum (%.2f s); resetting buffer.",
                 age.toSec(), max_buffer_age_);
        resetBuffer();
      }
    }
  }

  // Reset the frame buffer and all internal state tensors.
  void resetBuffer() {
    std::lock_guard<std::mutex> lock(frame_vector_mutex_);
    frame_vector_.clear();
    initial_reference_ = torch::Tensor();
    timestamps_last_event_ = torch::Tensor();
    last_image_ = torch::Tensor();
    last_timestamp_ = torch::Tensor();
  }

  // Preprocess image: convert to grayscale, resize, and normalize to [0,1] (no log transform).
  cv::Mat preprocess(const cv::Mat& img, const std::string& encoding) {
    cv::Mat gray;
    if (encoding == "mono8" || encoding == "8UC1") {
      gray = img.clone();
    } else if (encoding == "rgb8") {
      cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);
    } else if (encoding == "bgr8") {
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
      ROS_ERROR("Unsupported encoding: %s", encoding.c_str());
      return {};
    }
    if (gray.cols != width_ || gray.rows != height_) {
      cv::resize(gray, gray, cv::Size(width_, height_));
    }
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    return gray;
  }

  // Timer callback: process accumulated frames.
  void timerCallback(const ros::TimerEvent&) {
    // Optional: if frames in buffer are too old, reset buffer.
    checkBufferAge();

    std::vector<FrameData> frames;
    {
      std::lock_guard<std::mutex> lock(frame_vector_mutex_);
      if (frame_vector_.empty())
        return;
      frames.swap(frame_vector_);
    }

    // Build tensors for images [T x H x W] and timestamps [T].
    torch::TensorOptions fopts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    torch::TensorOptions iopts = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);

    int T_new = static_cast<int>(frames.size());
    torch::Tensor imgs = torch::empty({T_new, height_, width_}, fopts);
    torch::Tensor ts   = torch::empty({T_new}, iopts);

    for (int i = 0; i < T_new; i++) {
      auto frame_i = frames[i].frame;
      torch::Tensor img_tensor = torch::from_blob(frame_i.data, {height_, width_}, torch::kFloat32).clone();
      imgs[i] = img_tensor.to(torch::kCUDA);
      double t_sec = frames[i].timestamp.toSec();
      int64_t t_us = static_cast<int64_t>(t_sec * 1e6);
      ts[i] = t_us;
    }

    // If a previous frame exists, prepend it.
    if (last_image_.defined()) {
      imgs = torch::cat(std::vector<torch::Tensor>{last_image_, imgs}, 0);
      ts   = torch::cat(std::vector<torch::Tensor>{last_timestamp_, ts}, 0);
    }

    // If only one image is available, store it for next cycle and exit.
    if (imgs.size(0) == 1) {
      last_image_ = imgs.index({0}).unsqueeze(0);
      last_timestamp_ = ts.index({0}).unsqueeze(0);
      return;
    }

    // On first run, initialize reference and last-event timestamp tensors.
    if (!initial_reference_.defined()) {
      initial_reference_ = imgs[0].clone();
      timestamps_last_event_ = torch::zeros_like(initial_reference_).to(torch::kInt64);
    }

    int T_total = imgs.size(0);
    torch::Tensor refs_over_time = torch::zeros({T_total - 1, height_, width_}, fopts);
    torch::Tensor event_counts   = torch::zeros({height_, width_}, iopts);

    try {
      auto ret = esim_forward_count_events(imgs,
                                           initial_reference_,
                                           refs_over_time,
                                           event_counts,
                                           ct_neg_, ct_pos_);
      refs_over_time = ret[0];
      event_counts   = ret[1];
    } catch (const std::runtime_error &e) {
      ROS_ERROR("CUDA error in forward_count_events: %s", e.what());
      return;
    }

    torch::Tensor cumsum = event_counts.view({-1}).cumsum(0);
    int64_t total_num_events = cumsum[-1].item<int64_t>();
    torch::Tensor offsets = cumsum.view({height_, width_}) - event_counts;

    if (total_num_events == 0) {
      dv_ros_msgs::EventArray empty_msg;
      empty_msg.header.stamp = ros::Time::now();
      empty_msg.width  = width_;
      empty_msg.height = height_;
      event_pub_.publish(empty_msg);
      last_image_ = imgs.index({-1}).unsqueeze(0);
      last_timestamp_ = ts.index({-1}).unsqueeze(0);
      return;
    }

    torch::Tensor events = torch::zeros({total_num_events, 4}, iopts);

    try {
      events = esim_forward(imgs, ts,
                            initial_reference_,
                            refs_over_time,
                            offsets,
                            events,
                            timestamps_last_event_,
                            ct_neg_,
                            ct_pos_,
                            refractory_period_ns_);
    } catch (const std::runtime_error &e) {
      ROS_ERROR("CUDA error in forward: %s", e.what());
      return;
    }

    events = events.to(torch::kCPU);
    if (events.size(0) == 0) {
      dv_ros_msgs::EventArray empty_msg;
      empty_msg.header.stamp = ros::Time::now();
      empty_msg.width  = width_;
      empty_msg.height = height_;
      event_pub_.publish(empty_msg);
      last_image_ = imgs.index({-1}).unsqueeze(0);
      last_timestamp_ = ts.index({-1}).unsqueeze(0);
      return;
    }

    // Sort events by timestamp (column 2) and filter out any with non-positive timestamp.
    auto ts_col = events.select(1, 2);
    auto sort_result = ts_col.sort(0);
    auto sorted_indices = std::get<1>(sort_result);
    events = events.index_select(0, sorted_indices);
    auto valid_mask = events.select(1, 2) > 0;
    auto valid_indices = torch::nonzero(valid_mask).squeeze();
    if (valid_indices.dim() == 0)
      valid_indices = valid_indices.unsqueeze(0);
    events = events.index_select(0, valid_indices);

    // Update the internal reference state.
    initial_reference_ = refs_over_time.index({-1}).clone();
    last_image_ = imgs.index({-1}).unsqueeze(0);
    last_timestamp_ = ts.index({-1}).unsqueeze(0);

    // Build and publish the ROS EventArray message.
    dv_ros_msgs::EventArray msg;
    msg.header.stamp = ros::Time::now();
    msg.width  = width_;
    msg.height = height_;
    msg.events.reserve(events.size(0));
    auto ev_acc = events.accessor<int64_t, 2>();
    for (int64_t i = 0; i < events.size(0); i++) {
      dv_ros_msgs::Event e;
      e.x = ev_acc[i][0];
      e.y = ev_acc[i][1];
      int64_t t_us = ev_acc[i][2];
      if (t_us <= 0)
        continue;
      double t_sec = static_cast<double>(t_us) * 1e-6;
      e.ts = ros::Time(t_sec);
      e.polarity = (ev_acc[i][3] == 1) ? 1 : 0;
      msg.events.push_back(e);
    }
    event_pub_.publish(msg);
    ROS_INFO("Published %zu events.", msg.events.size());
  }

  // ROS members.
  ros::NodeHandle nh_;
  ros::Publisher event_pub_;
  ros::Subscriber image_sub_;
  ros::Subscriber collision_sub_;
  ros::Timer timer_;

  // Frame buffer.
  std::mutex frame_vector_mutex_;
  std::vector<FrameData> frame_vector_;

  // Parameters.
  int width_, height_;
  float ct_pos_, ct_neg_;
  int64_t refractory_period_ns_;
  double max_buffer_age_; // maximum allowed age for frames (in seconds); 0 disables check.

  // Simulator state.
  torch::Tensor initial_reference_;
  torch::Tensor timestamps_last_event_;
  torch::Tensor last_image_;
  torch::Tensor last_timestamp_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "esim_cuda_node");
  EventCameraCudaNode node;
  ros::spin();
  return 0;
}
