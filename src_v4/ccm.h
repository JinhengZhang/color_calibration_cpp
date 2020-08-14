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
        enum INITIAL_METHOD_TYPE {
            WHITE_BALANCE,
            LEAST_SQUARE
        };

        /* After being called, the method produce a ColorCorrectionModel instance for inference.*/
        class ColorCorrectionModel
        {
        public:
            // detected colors, the referenceand the RGB colorspace for conversion
            int shape;
            cv::Mat src;
            Color dst;

            // linear method
            RGB_Base_& cs;
            Linear* linear;

            // weights and mask
            cv::Mat weights;
            cv::Mat mask;
            int masked_len;

            // RGBl of detected data and the reference
            cv::Mat src_rgbl;
            cv::Mat dst_rgbl;

            DISTANCE_TYPE distance;

            cv::Mat dist;
            cv::Mat ccm;
            cv::Mat ccm0;

            double error;
            double maxCount;
            double epsilon;
            ColorCorrectionModel(cv::Mat src, Color dst, RGB_Base_ cs, DISTANCE_TYPE distance, LINEAR_TYPE linear,
                double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list, 
                double weights_coeff, INITIAL_METHOD_TYPE initial_method_type, double maxCount, double epsilon);

            // calculate weights and mask
            void _cal_weights_masks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask);
            
            // make no change for ColorCorrectionModel_3x3 class
            // convert matrix A to [A, 1] in ColorCorrectionModel_4x3 class
            virtual cv::Mat prepare(cv::Mat inp) { return inp; };
            
            // fitting nonlinear - optimization initial value by white balance :
            // res = diag(mean(s_r) / mean(d_r), mean(s_g) / mean(d_g), mean(s_b) / mean(d_b))
            // see CCM.pdf for details;
            cv::Mat initial_white_balance(void) ;
            
            // fitting nonlinear-optimization initial value by least square:
            // res = np.linalg.lstsq(src_rgbl, dst_rgbl)
            // see CCM.pdf for details;
            // if fit==True, return optimalization for rgbl distance function;
            
            cv::Mat initial_least_square(bool fit = false);
            
            // fitting ccm if distance function is associated with CIE Lab color space
            void fitting(void);

            // infer using fittingd ccm
            cv::Mat infer(cv::Mat img, bool L = false);

            // infer image and output as an BGR image with uint8 type
            // mainly for test or debug!
            cv::Mat infer_image(std::string imgfile, bool L = false, int inp_size = 255, int out_size = 255);//
        };


        class ColorCorrectionModel_3x3 : public ColorCorrectionModel 
        {
        public:
            
            ColorCorrectionModel_3x3(cv::Mat src, Color dst, RGB_Base_ cs, DISTANCE_TYPE distance, LINEAR_TYPE linear,
                double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list,double weights_coeff, 
                INITIAL_METHOD_TYPE initial_method_type, double maxCount, double epsilon) : ColorCorrectionModel(src, dst, cs, distance,
                    linear, gamma, deg, saturated_threshold, weights_list, weights_coeff, initial_method_type, maxCount, epsilon) {
                shape = 9;
            }
        };


        class ColorCorrectionModel_4x3 : public ColorCorrectionModel
        {
        public:
            ColorCorrectionModel_4x3(cv::Mat src, Color dst, RGB_Base_ cs, DISTANCE_TYPE distance, LINEAR_TYPE linear,
                double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list, double weights_coeff,
                INITIAL_METHOD_TYPE initial_method_type, double maxCount, double epsilon) : ColorCorrectionModel(src, dst, cs, distance,
                    linear, gamma, deg, saturated_threshold, weights_list, weights_coeff, initial_method_type, maxCount, epsilon) {
                shape = 12;
            }
            cv::Mat prepare(cv::Mat arr);
        };


        ColorCorrectionModel* color_correction(CCM_TYPE ccm_type, cv::Mat src, Color dst, RGB_Base_ cs, 
            DISTANCE_TYPE distance, LINEAR_TYPE linear, double gamma, int deg, std::vector<double> saturated_threshold,
            cv::Mat weights_list, double weights_coeff, INITIAL_METHOD_TYPE initial_method_type, double maxCount, double epsilon);
    }
}
#endif
