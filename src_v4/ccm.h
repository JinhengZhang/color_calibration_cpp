#ifndef ColorCorrectionModel_H
#define ColorCorrectionModel_H

#include<iostream>
#include<cmath>
#include<string>
#include<vector>
#include "utils.h"
#include "distance.h"
#include "linearize.h"
#include "colorspace.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"
#include "color.h"

namespace cv {
    namespace ccm {
        enum CCM_TYPE {
            CCM_3x3,
            CCM_4x3
        };

        class ColorCorrectionModel
        {
        public:
            int shape;
            cv::Mat src;
            Color dst;

            RGB_Base_& cs;
            Linear* linear;

            cv::Mat weights;
            cv::Mat mask;
            int masked_len;

            cv::Mat src_rgbl;
            cv::Mat dst_rgbl;

            std::string distance;

            cv::Mat dist;
            cv::Mat ccm;
            cv::Mat ccm0;

            double error;
            double xtol;
            double ftol;
            ColorCorrectionModel(cv::Mat src, Color dst, RGB_Base_ cs, std::string distance, LINEAR_TYPE linear,
                double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list, 
                double weights_coeff, std::string initial_method, double xtol, double ftol);

            void _cal_weights_masks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask);
            virtual cv::Mat prepare(cv::Mat inp) { return inp; };
            cv::Mat initial_white_balance(void) ;
            cv::Mat initial_least_square(bool fit = false);
            void fitting(void);
            cv::Mat infer(cv::Mat img, bool L = false);
            cv::Mat infer_image(std::string imgfile, bool L = false, int inp_size = 255, int out_size = 255);//
        };


        class ColorCorrectionModel_3x3 : public ColorCorrectionModel 
        {
        public:
            
            ColorCorrectionModel_3x3(cv::Mat src, Color dst, RGB_Base_ cs, std::string distance, LINEAR_TYPE linear,
                double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list,double weights_coeff, 
                std::string initial_method, double xtol, double ftol) : ColorCorrectionModel(src, dst, cs, distance, 
                    linear, gamma, deg, saturated_threshold, weights_list, weights_coeff, initial_method, xtol, ftol) {
                shape = 9;
            }
        };


        class ColorCorrectionModel_4x3 : public ColorCorrectionModel
        {
        public:
            ColorCorrectionModel_4x3(cv::Mat src, Color dst, RGB_Base_ cs, std::string distance, LINEAR_TYPE linear,
                double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list, double weights_coeff,
                std::string initial_method, double xtol, double ftol) : ColorCorrectionModel(src, dst, cs, distance,
                    linear, gamma, deg, saturated_threshold, weights_list, weights_coeff, initial_method, xtol, ftol) {
                shape = 12;
            }
            cv::Mat prepare(cv::Mat arr);
        };


        ColorCorrectionModel* color_correction(CCM_TYPE ccm_type, cv::Mat src, Color dst, RGB_Base_ cs, 
            std::string distance, LINEAR_TYPE linear, double gamma, int deg, std::vector<double> saturated_threshold, 
            cv::Mat weights_list, double weights_coeff, std::string initial_method, double xtol, double ftol);
    }
}
#endif
