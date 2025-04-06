#ifndef CDDFUSE_HPP
#define CDDFUSE_HPP

#include <iostream>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>


namespace CDDFUSE {
    using namespace std;

    class Fuser{
    public:
        virtual cv::Mat fuse(cv::Mat& img_rgb, cv::Mat& img_tir) = 0;
    };

    shared_ptr<Fuser> create_fuser(const std::string &engine_path, int gpuid = 0);
}


#endif //CDDFUSE_HPP
