#ifndef CCM_H
#define CCM_H

#include<iostream>
#include<cmath>
#include "utils.h"
#include "distance.h"
#include "linearize.h"
#include "colorspace.h"
#include "colorchecker.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv;


class CCM_3x3
{
public:
   
    Mat src;
    ColorCheckerMetric cc;
    RGB_Base* cs;    
    Linear* linear;
    Mat weights;
    vector<bool> mask;
    Mat src_rgbl;
    Mat src_rgb_masked;
    Mat src_rgbl_masked;
    Mat dst_rgb_masked;
    Mat dst_rgbl_masked;
    Mat dst_lab_masked;
    Mat weights_masked;
    Mat weights_masked_norm;
    int masked_len;
    string distance;
    double xtol;
    double ftol;
    Mat dist;
    Mat ccm;
    Mat ccm0;

    /*struct color_c {
       Mat dst;
       string dst_colorspace;
       string dst_illuminant;
       string dst_observer;
       int dst_whites;
       string colorchecker;
       string ccm_shape;
       float* saturated_threshold;
       string colorspace;
       string linear;
       float gamma;
       float deg;
       string distance;
       string dist_illuminant;
       string dist_observer;
       Mat weights_list;
       float weights_coeff;
       bool weights_color;
       string initial_method;
       float xtol;
       float ftol;
    } color_ca;*/
    //struct color_c* pcolor_c = &color_ca;
    //CCM_3x3(Mat src_, struct color_c* pcolor_c);

    CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, string ccm_shape, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, float deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method, double xtol_, double ftol_);

    void prepare(void) {};
    Mat initial_white_balance(Mat src_rgbl, Mat dst_rgbl);
    Mat initial_least_square(Mat src_rgbl, Mat dst_rgbl);
    double loss_rgb(Mat ccm);
    void calculate_rgb(void);
    double loss_rgbl(Mat ccm);
    void calculate_rgbl(void);
    double loss(Mat ccm);
    void calculate(void);
    void value(int number);
    Mat infer(Mat img, bool L);
    Mat infer_image(string imgfile, bool L, int inp_size, int out_size, string out_dtype);
};


class CCM_4x3 : public CCM_3x3
{
public:
    using CCM_3x3::CCM_3x3;

    void prepare(void) {};
    Mat add_column(Mat arr) {};
    Mat initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {};
    Mat infer(Mat img, bool L) {};
    void value(int number) {};
};


#endif
