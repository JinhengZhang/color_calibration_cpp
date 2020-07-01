#ifndef CCM_H
#define CCM_H


#include<iostream>
#include<cmath>
#include "utils.h"
#include "colorspace.h"
#include "colorchecker.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv;


#define COLORSPACE(colorspace) colorspace() 
#define COLORCHECKER(colorchecker) ColorChecker_##colorchecker() 
#define LINEAR(linear) Linear_##linear(float gamma_, int deg_, Mat src_, ColorCheckerMetric colorchecker_,  int *saturated_threshold_) 
#define DISTANCE(distance) distance_##distance() 
#define INITIAL(initial_method) initial_##initial_method


class CCM_3x3
{
public:
    struct color_c {
        Mat dst;
        char dst_colorspace;
        char dst_illuminant;
        char dst_observer;
        int dst_whites;
        char colorchecker;
        char ccm_shape;
        float* saturated_threshold;
        char colorspace;
        char linear;
        float gamma;
        float deg;
        char distance;
        char dist_illuminant;
        char dist_observer;
        Mat weights_list;
        float weights_coeff;
        bool weights_color;
        char initial_method;
        float xtol;
        float ftol;
    } color_ca;
    Mat src;
    struct color_c* pcolor_c = &color_ca;

    CCM_3x3(Mat src_, struct color_c* pcolor_c);

    void prepare(void);
    Mat initial_white_balance(Mat src_rgbl, Mat dst_rgbl);
    Mat initial_least_square(Mat src_rgbl, Mat dst_rgbl);
    float loss_rgb(Mat ccm);
    void calculate_rgb(void);
    float loss_rgbl(Mat ccm);
    void calculate_rgbl(void);
    float loss(Mat ccm);
    void calculate(void);
    void value(int number);
    Mat infer(Mat img, bool L);
    Mat infer_image(char imgfile, bool L, int inp_size, int out_size, char out_dtype);
};


class CCM_4x3 : public CCM_3x3
{
public:
    void prepare(void) {};
    Mat add_column(Mat arr) {};
    Mat initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {};
    Mat infer(Mat img, bool L) {};
    void value(int number) {};
};


#endif
