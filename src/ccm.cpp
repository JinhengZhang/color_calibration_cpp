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

CCM_3x3(Mat src_, struct color_c* pcolor_c) {
    src = src_;
    dist_io = IO(dist_illuminant_, dist_observer_);
    cs = COLORSPACE(colorspace);
    cs.set_default(dist_io);
    if (dst) { 
        cc_ = ColorChecker(dst_, dst_colorspace_, IO(dst_illuminant_, dst_observer_), dst_whites_); 
    }
    else { 
        cc_ = COLORCHECKER(colorchecker); 
    }
    cc = ColorCheckerMetric(cc_, cs, dist_io);
    linear = LINEAR(linear);
    weight = NULL;
    if (weights_list) { 
        weights = weights_list_;
    }
    elif (weights_coeff!=0){ 
        Mat cc_lab_0 = cc.lab.rowRange(0, 1).clone();
        weights = pow(cc_lab_0, weights_coeff_);
    }

    weight_mask_ = Mat::ones(1, src.rows, cv_8UC1, bool);
    if (weights_color_)
    {
        weight_mask_ = cc.color_mask;
    }

    saturate_mask = saturate(src, saturated_threshold)
    mask = saturate_mask & weight_mask
    src_rgbl = linear.linearize(src)
    src_rgb_masked = src[mask]
    src_rgbl_masked = src_rgbl[mask]
    dst_rgb_masked = cc.rgb[mask]
    dst_rgbl_masked = cc.rgbl[mask]
    dst_lab_masked = cc.lab[mask]
    if (!weights.data)
    {
        weights_masked = weights[mask];
        weights_masked_norm = weights_masked / np.mean(weights_masked);
    }
    masked_len = src_rgb_masked.rows
    distance = DISTANCE(distance_);
    inital_func = INITIAL(initial_method_);
    xtol = xtol_
    ftol = ftol_
    ccm = NULL
    if (distance == "rgb")
    {
        calculate_rgb();
    }
    elif(distance == "rgbl")
    {
        calculate_rgbl();
    }
    else
    {
        calculate();
    }

}

void CCM_3x3::prepare(void) {
}

Mat CCM_3x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    rs = sum(src_rgbl.rowRange(0, 1).clone());
    gs = sum(src_rgbl.rowRange(1, 2).clone());
    bs = sum(src_rgbl.rowRange(2, 3).clone());
    rd = sum(src_rgbl.rowRange(0, 1).clone());
    gd = sum(src_rgbl.rowRange(1, 2).clone());
    bd = sum(src_rgbl.rowRange(2, 3).clone());
    Mat initial_white_balance_ = (Mat_<float>(3, 3) << rd / rs, 0, 0, 0, gd / gs, 0, 0, 0, bd / bs);
    return initial_white_balance_;
}

Mat CCM_3x3::initial_least_square(Mat src_rgbl, Mat dst_rgbl) {
    return src_rgbl.inv() * dst_rgbl;
}

float CCM_3x3::loss_rgb(Mat ccm) {
    ccm = ccm.reshape(0, 3);
    lab_est = cs.rgbl2rgb(src_rgbl_masked * ccm);
    dist = distance(lab_est, dst_rgb_masked);
    dist = pow(dist, 2);
    if (weights.data)
    {
        dist = weights_masked_norm * dist;
    }
    return sum(dist);
}

void CCM_3x3::calculate_rgb(void) {
    ccm0 = inital_func(src_rgbl_masked, dst_rgbl_masked);
    ccm0 = ccm0.reshape(0, 1);
    //res = fmin(loss_rgb, ccm0, xtol = xtol, ftol = ftol);
    if (res.data)
    {
        ccm = res.reshape(0, 3);
        error = pow((loss_rgb(res) / masked_len), 0.5);
        cout << 'ccm' << ccm << endl;
        cout << 'error:' << error << endl;
    }
}

float CCM_3x3::loss_rgbl(Mat ccm) {
    dist = sum(power(dst_rgbl_masked - src_rgbl_masked * ccm, 2));
    if (weights.data)
    {
        dist = weights_masked_norm * dist;
    }
    return sum(dist);
}

void CCM_3x3::calculate_rgbl(void) {
    if (weights.data)
    {
        ccm = initial_least_square(src_rgbl_masked, dst_rgbl_masked);
    }
    else
    {
        //w = np.diag(pow(weights_masked_norm, 0.5));
        ccm = initial_least_square(src_rgbl_masked * w, dst_rgbl_masked * w);
    }
    error = pow((loss_rgbl(ccm) / masked_len), 0.5);
}

float CCM_3x3::loss(Mat ccm) {
    ccm = ccm.reshape(0, 3);
    lab_est = cs.rgbl2lab(src_rgbl_masked * ccm);
    dist = distance(lab_est, dst_lab_masked);
    dist = pow(dist, 2);
    if (weights.data)
    {
        dist = weights_masked_norm * dist;
    }
    return sum(dist);
}

void CCM_3x3::calculate(void) {
    ccm0 = inital_func(src_rgbl_masked, dst_rgbl_masked);
    ccm0 = ccm0.reshape(0, 1);
    //res = fmin(loss_rgb, ccm0, xtol = xtol, ftol = ftol);
    if (res.data)
    {
        ccm = res.reshape(0, 3);
        error = pow((loss(res) / masked_len), 0.5);
        cout << "ccm" << ccm << endl;
        cout << "error:" << error << endl;
    }
}

void CCM_3x3::value(int number) {
    cout << "error:" << error << endl;
    //rand = np.random.random((number, 3));
    mask = saturate(infer(rand), 0, 1);
    sat = sum(mask) / number;
    cout << "sat:" << sat << endl;
    rgbl = cs.rgb2rgbl(rand);
    //mask = saturate(rgbl * np.linalg.inv(self.ccm), 0, 1);
    dist = sum(mask) / number;
    cout << "dist:" << dist << endl;
}

Mat CCM_3x3::infer(Mat img, bool L) {
    if (!ccm.data)
    {
        throw "No CCM values!";
    }
    img_lin = linear.linearize(img)
    img_ccm = img_lin * ccm
    if (L)
    {
        return img_ccm; 
    }
    return cs.rgbl2rgb(img_ccm);
}

Mat CCM_3x3::infer_image(char imgfile, bool L, int inp_size, int out_size, char out_dtype) {
    img = imread(imgfile)
    img = cvtColor(img, img, CV_BGR2RGB) / inp_size
    out = infer(img, L)
    //img = np.minimum(np.maximum(np.round(out * out_size), 0), out_size)
    //img = img.astype(out_dtype)
    return cvtColor(img, img, CV_RGB2BGR);
}



class CCM_4x3 : public CCM_3x3
{
public:
    void prepare(void) {};
    Mat add_column(Mat arr) {};
    Mat initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {};
    Mat infer(Mat img, bool L) {};
    void value(int number) {};
};

void CCM_4x3::prepare(void) {
    src_rgbl_masked = add_column(src_rgbl_masked);
}

Mat CCM_4x3::add_column(Mat arr) {
    //return np.c_[arr, np.ones((*arr.shape[:-1], 1))];
}

Mat CCM_4x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    rs = sum(src_rgbl.rowRange(0, 1).clone());
    gs = sum(src_rgbl.rowRange(1, 2).clone());
    bs = sum(src_rgbl.rowRange(2, 3).clone());
    rd = sum(src_rgbl.rowRange(0, 1).clone());
    gd = sum(src_rgbl.rowRange(1, 2).clone());
    bd = sum(src_rgbl.rowRange(2, 3).clone());
    Mat initial_white_balance_ = (Mat_<float>(3, 3) << rd / rs, 0, 0, 0, gd / gs, 0, 0, 0, bd / bs);
    return initial_white_balance_;
}

Mat CCM_4x3::infer(Mat img, bool L) {
    if (!ccm.data) {
        throw "No CCM values!";
    }
    img_lin = linear.linearize(img);
    img_ccm = add_column(img_lin) * ccm;
    if (L) {
        return img_ccm;
    }
    return cs.rgbl2rgb(img_ccm);
}

void CCM_4x3::value(int number) {
    cout << "error:" << error << endl;
    //rand = np.random.random((number, 3));
    mask = saturate(infer(rand), 0, 1);
    sat = sum(mask) / number;
    cout << "sat:" << sat << endl;
    rgbl = cs.rgb2rgbl(rand);
    //up, down = self.ccm[:3, : ], self.ccm[3:, : ]
    //mask = saturate((rgbl - np.ones((number, 1))@down)@np.linalg.inv(up), 0, 1)
    dist = sum(mask) / number;
    cout << "dist:" << dist << endl;
}
