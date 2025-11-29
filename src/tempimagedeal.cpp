#include "image_deal.h"
#include "shape_tools.h"

#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

Logger logger;

struct DetectedObject
{
  std::string type;
  std::vector<cv::Point2f> points;
};

// void vision_node::callback_stage(referee_pkg::msg::RaceStage)
// {
//   this->latest_stage = stage;
// }

void vision_node::callback_camera(sensor_msgs::msg::Image::SharedPtr msg)
{
  try
  {
    // 图像转换：从ROS的Img到opencv的Mat
    cv_bridge::CvImagePtr cv_ptr;
    if (msg->encoding == "rgb8" || msg->encoding == "R8G8B8")
    {
      cv::Mat image(msg->height, msg->width, CV_8UC3,
                    const_cast<unsigned char *>(msg->data.data()));
      cv::Mat bgr_image;
      cv::cvtColor(image, bgr_image, cv::COLOR_RGB2BGR);
      cv_ptr = std::make_shared<cv_bridge::CvImage>();
      cv_ptr->header = msg->header;
      cv_ptr->encoding = "bgr8";
      cv_ptr->image = bgr_image;
    }
    else
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }

    cv::Mat image = cv_ptr->image;

    if (image.empty())
    {
      RCLCPP_WARN(this->get_logger(), "Received empty image");
      return;
    }

    std::vector<DetectedObject> all_detected_objects;

    std::vector<std::string> point_names = {"#1#", "#2#", "#3#", "#4#"};
    std::vector<cv::Scalar> point_colors = {
        cv::Scalar(255, 0, 0),   // 蓝色 - 1
        cv::Scalar(0, 255, 0),   // 绿色 - 2
        cv::Scalar(0, 255, 255), // 黄色 - 3
        cv::Scalar(255, 0, 255)  // 紫色 - 4
    };

    // 创建结果图像
    cv::Mat result_image = image.clone();

    // 转换到 HSV 空间
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    /*===========================red_ring===========================zp*/
    // 红色检测 - 使用稳定的范围
    cv::Mat mask1, mask2, red_mask;
    cv::inRange(hsv, sphere_red_low1, sphere_red_high1, mask1);
    cv::inRange(hsv, sphere_red_low2, sphere_red_high2, mask2);
    red_mask = mask1 | mask2;

    // 适度的形态学操作
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(red_mask, red_mask, cv::MORPH_OPEN, kernel);

    // 找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(red_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 统计球的数量
    int valid_spheres = 0;

    for (size_t i = 0; i < contours.size(); i++)
    {
      double area = cv::contourArea(contours[i]);
      if (area < 500)
        continue;

      // 计算最小外接圆
      cv::Point2f center;
      float radius = 0;
      cv::minEnclosingCircle(contours[i], center, radius);

      // 计算圆形度
      double perimeter = cv::arcLength(contours[i], true);
      double circularity = 4 * CV_PI * area / (perimeter * perimeter);

      if (circularity > 0.7 && radius > 15 && radius < 200)
      {
        valid_spheres++;

        // 求出四个点坐标
        std::vector<cv::Point2f> sphere_points =
            shape_tools::calculateStableSpherePoints(center, radius);

        RCLCPP_INFO(this->get_logger(), "Found sphere %d: (%.1f, %.1f) R=%.1f C=%.3f",
                    valid_spheres, center.x, center.y, radius, circularity);

        // 绘制检测到的球体
        cv::circle(result_image, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 2); // 绿色圆圈
        cv::circle(result_image, center, 3, cv::Scalar(0, 0, 255), -1);                       // 红色圆心

        // 绘制球体上的四个点
        for (int j = 0; j < 4; j++)
        {
          cv::circle(result_image, sphere_points[j], 6, point_colors[j], -1);
          cv::circle(result_image, sphere_points[j], 6, cv::Scalar(0, 0, 0), 2);

          // 标注序号
          std::string point_text = std::to_string(j + 1);
          cv::putText(
              result_image, point_text,
              cv::Point(sphere_points[j].x + 5, sphere_points[j].y - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
          cv::putText(
              result_image, point_text,
              cv::Point(sphere_points[j].x + 5, sphere_points[j].y - 5),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

          RCLCPP_INFO(this->get_logger(), "Sphere %d, Point(%s): (%.1f, %.1f)",
                      valid_spheres, point_names[j].c_str(), sphere_points[j].x, sphere_points[j].y);
        }

        // 添加到发送列表
        DetectedObject sphere_obj;
        sphere_obj.type = "Ring_red";
        sphere_obj.points = sphere_points;
        all_detected_objects.push_back(sphere_obj);

        double small_radius = 0.68 * radius;

        if (circularity > 0.7 && radius > 15 && radius < 200)
        {
          valid_spheres++;

          // 求出四个点坐标
          std::vector<cv::Point2f> small_sphere_points =
              shape_tools::calculateStableSpherePoints(center, small_radius);

          RCLCPP_INFO(this->get_logger(), "Found sphere %d: (%.1f, %.1f) R=%.1f",
                      valid_spheres, center.x, center.y, small_radius);

          // 绘制检测到的球体
          cv::circle(result_image, center, static_cast<int>(small_radius), cv::Scalar(255, 0, 0), 2); // 蓝色圆圈

          // 绘制球体上的四个点
          for (int j = 0; j < 4; j++)
          {
            cv::circle(result_image, small_sphere_points[j], 6, point_colors[j], -1);
            cv::circle(result_image, small_sphere_points[j], 6, cv::Scalar(0, 0, 0), 2);

            // 标注序号
            std::string point_text = std::to_string(j + 1);
            cv::putText(
                result_image, point_text,
                cv::Point(small_sphere_points[j].x + 5, small_sphere_points[j].y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 3);
            cv::putText(
                result_image, point_text,
                cv::Point(small_sphere_points[j].x + 5, small_sphere_points[j].y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, point_colors[j], 2);

            RCLCPP_INFO(this->get_logger(), "Sphere %d, Point(%s): (%.1f, %.1f)",
                        valid_spheres, point_names[j].c_str(), small_sphere_points[j].x, small_sphere_points[j].y);
          }

          // 添加到发送列表
          DetectedObject sphere_obj;
          sphere_obj.type = "Ring_red";
          sphere_obj.points = small_sphere_points;
          all_detected_objects.push_back(sphere_obj);
        }
      }
    }

    /*===========================Arrow===========================ztw*/
    cv::Mat arrowred_mask;
    arrowred_mask = mask1 | mask2;

    // 找轮廓
    std::vector<std::vector<cv::Point>> arrowred_contours;
    cv::findContours(arrowred_mask, arrowred_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // 只保留最像箭头的那个轮廓（面积大且细长）
    int best_idx = -1;
    double best_score = 0.0; // 这里用 ratio 做一个简单打分

    for (size_t i = 0; i < arrowred_contours.size(); i++)
    {
      double area = cv::contourArea(arrowred_contours[i]);
      if (area < 50.0) // 太小的噪声过滤掉，阈值按需要调
        continue;

      cv::RotatedRect rr = cv::minAreaRect(arrowred_contours[i]);
      float w = rr.size.width;
      float h = rr.size.height;
      if (w < 5 || h < 5) // 太细太小的也过滤
        continue;

      float long_side = std::max(w, h);
      float short_side = std::min(w, h);
      float ratio = long_side / short_side; // 长宽比

      if (ratio < 1.5f) // 箭头一般比较细长，长宽比至少大于 2，视情况调
         continue;

      double score = ratio; // 细长程度兼顾
      if (score > best_score)
      {
        best_score = score;
        best_idx = static_cast<int>(i);
      }
    }

    int valid_arrows = 0;

    if (best_idx >= 0)
    {
      valid_arrows++;

      const auto &cnt = arrowred_contours[best_idx];
      cv::RotatedRect rr = cv::minAreaRect(cnt);
      float w = rr.size.width;
      float h = rr.size.height;
      float ratio = (w > h ? w / h : h / w);

      // minAreaRect 的 4 个角点
      cv::Point2f rr_pts[4];
      rr.points(rr_pts);
      std::vector<cv::Point2f> pts(rr_pts, rr_pts + 4);

      // 排序成：TL TR BR BL
      std::sort(pts.begin(), pts.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.y < b.y; });

      std::vector<cv::Point2f> top{pts[0], pts[1]};
      std::vector<cv::Point2f> bottom{pts[2], pts[3]};

      std::sort(top.begin(), top.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.x < b.x; });
      std::sort(bottom.begin(), bottom.end(),
                [](const cv::Point2f &a, const cv::Point2f &b)
                { return a.x < b.x; });

      cv::Point2f tl = top[0], tr = top[1];
      cv::Point2f bl = bottom[0], br = bottom[1];

      std::vector<cv::Point2f> arrow_points = {tl, tr, br, bl};

      // 画外接矩形
      for (int j = 0; j < 4; j++)
      {
        cv::line(result_image,
                 arrow_points[j],
                 arrow_points[(j + 1) % 4],
                 cv::Scalar(125, 125, 125),
                 4);
      }

      // 画点和编号
      for (int j = 0; j < 4; j++)
      {
        cv::circle(result_image, arrow_points[j], 7, point_colors[j], -1);
        cv::circle(result_image, arrow_points[j], 7, cv::Scalar(255, 255, 255), 2);

        std::string point_text = std::to_string(j + 1);
        cv::Point text_pos(arrow_points[j].x + 10, arrow_points[j].y - 10);

        cv::putText(result_image, point_text, text_pos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 255, 255), 4);
        cv::putText(result_image, point_text, text_pos,
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    point_colors[j], 2);

        RCLCPP_INFO(this->get_logger(),
                    "Arrow_0 %d, Point(%s): (%.1f, %.1f)",
                    valid_arrows, point_names[j].c_str(),
                    arrow_points[j].x, arrow_points[j].y);
      }

      RCLCPP_INFO(this->get_logger(),
                  "Found Arrow %d: center=(%.1f,%.1f) w=%.1f h=%.1f ratio=%.2f angle=%.1f",
                  valid_arrows, rr.center.x, rr.center.y,
                  w, h, ratio, rr.angle);

      DetectedObject arrow_obj;
      arrow_obj.type = "arrow";
      arrow_obj.points = arrow_points;
      all_detected_objects.push_back(arrow_obj);
    }
    /*=============================================================*/

    // 显示结果图像
    cv::imshow("Detection Result", result_image);
    cv::waitKey(1);

    // 创建并发布消息
    referee_pkg::msg::MultiObject msg_object;
    msg_object.header = msg->header;
    msg_object.num_objects = all_detected_objects.size();

    for (const auto &detected_obj : all_detected_objects)
    {
      referee_pkg::msg::Object obj_msg;

      // 放入目标类型
      obj_msg.target_type = detected_obj.type;

      // 放入目标四个点坐标
      for (const auto &point : detected_obj.points)
      {
        geometry_msgs::msg::Point corner;
        corner.x = point.x;
        corner.y = point.y;
        corner.z = 0.0;
        obj_msg.corners.push_back(corner);
      }

      // 放入单个目标信息
      msg_object.objects.push_back(obj_msg);
    }

    Target_pub->publish(msg_object);
    RCLCPP_INFO(this->get_logger(), "Published %lu total targets", all_detected_objects.size());
  }
  catch (const cv_bridge::Exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(this->get_logger(), "Exception: %s", e.what());
  }
}