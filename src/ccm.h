#ifndef CCM_H
#define CCM_H

#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"
#include "linearize.h"

using namespace std;
using namespace cv;

class CCM_3x3
{
public:
    int shape;
    cv::Mat src;
    ColorCheckerMetric cc;
    RGBBase* cs;
    Linear* linear;
    cv::Mat weights;
    cv::Mat mask;
    cv::Mat src_rgbl;
    cv::Mat src_rgb_masked;
    cv::Mat src_rgbl_masked;
    cv::Mat dst_rgb_masked;
    cv::Mat dst_rgbl_masked;
    cv::Mat dst_lab_masked;
    cv::Mat weights_masked;
    cv::Mat weights_masked_norm;
    int masked_len;
    string distance;
    cv::Mat dist;
    cv::Mat ccm;
    cv::Mat ccm0;

    /*struct color_c {
       cv::Mat dst;
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
       cv::Mat weights_list;
       float weights_coeff;
       bool weights_color;
       string initial_method;
       float xtol;
       float ftol;
    } color_ca;*/
    //struct color_c* pcolor_c = &color_ca;
    //CCM_3x3(cv::Mat src_, struct color_c* pcolor_c);
    CCM_3x3() {};
    CCM_3x3(cv::Mat src_, cv::Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, 
        cv::Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, 
        string linear_, double gamma, int deg, string distance_, string dist_illuminant, int dist_observer, 
        cv::Mat weights_list, double weights_coeff, bool weights_color, string initial_method, string shape);

    virtual void prepare(void) {}
    cv::Mat initial_white_balance(cv::Mat src_rgbl, cv::Mat dst_rgbl);
    cv::Mat initial_least_square(cv::Mat src_rgbl, cv::Mat dst_rgbl);
    void calculate_rgb(void);
    double loss_rgbl(cv::Mat ccm);
    void calculate_rgbl(void);
    void calculate(void);
    void value(int number);
    virtual cv::Mat infer(cv::Mat img, bool L=false);
    cv::Mat infer_image(string imgfile, bool L=false, int inp_size=255, int out_size=255);
    void calc(string initial_method, string distance_);
};

class CCM_4x3 : public CCM_3x3
{
public:
    using CCM_3x3::CCM_3x3;
    void prepare(void);
    cv::Mat add_column(cv::Mat arr);
    cv::Mat initial_white_balance(cv::Mat src_rgbl, cv::Mat dst_rgbl);
    cv::Mat infer(cv::Mat img, bool L);
    void value(int number);
};

static cv::Mat ColorChecker2005_LAB_D50_2 = (cv::Mat_<Vec3d>(24, 1) <<
    Vec3d(37.986, 13.555, 14.059),
    Vec3d(65.711, 18.13, 17.81),
    Vec3d(49.927, -4.88, -21.925),
    Vec3d(43.139, -13.095, 21.905),
    Vec3d(55.112, 8.844, -25.399),
    Vec3d(70.719, -33.397, -0.199),
    Vec3d(62.661, 36.067, 57.096),
    Vec3d(40.02, 10.41, -45.964),
    Vec3d(51.124, 48.239, 16.248),
    Vec3d(30.325, 22.976, -21.587),
    Vec3d(72.532, -23.709, 57.255),
    Vec3d(71.941, 19.363, 67.857),
    Vec3d(28.778, 14.179, -50.297),
    Vec3d(55.261, -38.342, 31.37),
    Vec3d(42.101, 53.378, 28.19),
    Vec3d(81.733, 4.039, 79.819),
    Vec3d(51.935, 49.986, -14.574),
    Vec3d(51.038, -28.631, -28.638),
    Vec3d(96.539, -0.425, 1.186),
    Vec3d(81.257, -0.638, -0.335),
    Vec3d(66.766, -0.734, -0.504),
    Vec3d(50.867, -0.153, -0.27),
    Vec3d(35.656, -0.421, -1.231),
    Vec3d(20.461, -0.079, -0.973));

static cv::Mat ColorChecker2005_LAB_D65_2 = (cv::Mat_<Vec3d>(24, 1) <<
    Vec3d(37.542, 12.018, 13.33),
    Vec3d(65.2, 14.821, 17.545),
    Vec3d(50.366, -1.573, -21.431),
    Vec3d(43.125, -14.63, 22.12),
    Vec3d(55.343, 11.449, -25.289),
    Vec3d(71.36, -32.718, 1.636),
    Vec3d(61.365, 32.885, 55.155),
    Vec3d(40.712, 16.908, -45.085),
    Vec3d(49.86, 45.934, 13.876),
    Vec3d(30.15, 24.915, -22.606),
    Vec3d(72.438, -27.464, 58.469),
    Vec3d(70.916, 15.583, 66.543),
    Vec3d(29.624, 21.425, -49.031),
    Vec3d(55.643, -40.76, 33.274),
    Vec3d(40.554, 49.972, 25.46),
    Vec3d(80.982, -1.037, 80.03),
    Vec3d(51.006, 49.876, -16.93),
    Vec3d(52.121, -24.61, -26.176),
    Vec3d(96.536, -0.694, 1.354),
    Vec3d(81.274, -0.61, -0.24),
    Vec3d(66.787, -0.647, -0.429),
    Vec3d(50.872, -0.059, -0.247),
    Vec3d(35.68, -0.22, -1.205),
    Vec3d(20.475, 0.049, -0.972));

static cv::Mat Arange_18_24 = (cv::Mat_<double>(1, 6) << 18, 19, 20, 21, 22, 23);
static ColorChecker colorchecker_Macbeth = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", D50_2, Arange_18_24);
static ColorChecker colorchecker_Macbeth_D65_2 = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", D65_2, Arange_18_24);

#endif
