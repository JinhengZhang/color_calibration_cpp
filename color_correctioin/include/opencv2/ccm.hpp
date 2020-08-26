#ifndef COLORCORRECTIONMODEL_H
#define COLORCORRECTIONMODEL_H


#include<iostream>
#include<cmath>
#include<string>
#include<vector>
#include "opencv2/ccm/linearize.hpp"


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
            int maxCount;
            double epsilon;
            ColorCorrectionModel(cv::Mat src, Color dst, RGB_Base_& cs, DISTANCE_TYPE distance,
                LINEAR_TYPE linear, double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list,
                double weights_coeff, INITIAL_METHOD_TYPE initial_method_type, int maxCount, double epsilon, int shape);

            // calculate weights and mask
            void _cal_weights_masks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask) ;

            // make no change for ColorCorrectionModel_3x3 class
            // convert matrix A to [A, 1] in ColorCorrectionModel_4x3 class
            cv::Mat prepare(cv::Mat inp);

            // fitting nonlinear - optimization initial value by white balance :
            // res = diag(mean(s_r) / mean(d_r), mean(s_g) / mean(d_g), mean(s_b) / mean(d_b))
            // see CCM.pdf for details;
            cv::Mat initial_white_balance(void);

            // fitting nonlinear-optimization initial value by least square:
            // res = np.linalg.lstsq(src_rgbl, dst_rgbl)
            // see CCM.pdf for details;
            // if fit==True, return optimalization for rgbl distance function;

            void initial_least_square(bool fit = false);


            // fitting ccm if distance function is associated with CIE Lab color space
            void fitting(void);

            cv::Mat infer(cv::Mat img, bool L = false);

            // infer image and output as an BGR image with uint8 type
            // mainly for test or debug!
            cv::Mat infer_image(std::string imgfile, bool L = false, int inp_size = 255, int out_size = 255);//
        };


       
    }
}
#endif
