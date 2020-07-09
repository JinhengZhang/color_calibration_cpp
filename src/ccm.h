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
    CCM_3x3() {};
    CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, float deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method, double xtol_, double ftol_);

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

    void prepare(void) ;
    Mat add_column(Mat arr) ;
    Mat initial_white_balance(Mat src_rgbl, Mat dst_rgbl) ;
    Mat infer(Mat img, bool L) ;
    void value(int number) ;
};

Mat ColorChecker2005_LAB_D50_2 = (Mat_<double>(24, 3) <<
    37.986, 13.555, 14.059,
    65.711, 18.13, 17.81,
    49.927, -4.88, -21.925,
    43.139, -13.095, 21.905,
    55.112, 8.844, -25.399,
    70.719, -33.397, -0.199,
    62.661, 36.067, 57.096,
    40.02, 10.41, -45.964,
    51.124, 48.239, 16.248,
    30.325, 22.976, -21.587,
    72.532, -23.709, 57.255,
    71.941, 19.363, 67.857,
    28.778, 14.179, -50.297,
    55.261, -38.342, 31.37,
    42.101, 53.378, 28.19,
    81.733, 4.039, 79.819,
    51.935, 49.986, -14.574,
    51.038, -28.631, -28.638,
    96.539, -0.425, 1.186,
    81.257, -0.638, -0.335,
    66.766, -0.734, -0.504,
    50.867, -0.153, -0.27,
    35.656, -0.421, -1.231,
    20.461, -0.079, -0.973);

Mat ColorChecker2005_LAB_D65_2 = (Mat_<double>(24, 3) <<
    37.542, 12.018, 13.33,
    65.2, 14.821, 17.545,
    50.366, -1.573, -21.431,
    43.125, -14.63, 22.12,
    55.343, 11.449, -25.289,
    71.36, -32.718, 1.636,
    61.365, 32.885, 55.155,
    40.712, 16.908, -45.085,
    49.86, 45.934, 13.876,
    30.15, 24.915, -22.606,
    72.438, -27.464, 58.469,
    70.916, 15.583, 66.543,
    29.624, 21.425, -49.031,
    55.643, -40.76, 33.274,
    40.554, 49.972, 25.46,
    80.982, -1.037, 80.03,
    51.006, 49.876, -16.93,
    52.121, -24.61, -26.176,
    96.536, -0.694, 1.354,
    81.274, -0.61, -0.24,
    66.787, -0.647, -0.429,
    50.872, -0.059, -0.247,
    35.68, -0.22, -1.205,
    20.475, 0.049, -0.972);


Mat Arange_18_24 = (Mat_<int>(1, 7) << 18, 19, 20, 21, 22, 23, 24);

ColorChecker colorchecker_Macbeth = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", IO("D65", 2), Arange_18_24);
ColorChecker colorchecker_Macbeth_D65_2 = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", IO("D65", 2), Arange_18_24);


#endif
