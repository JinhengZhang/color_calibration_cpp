#include "linearize.h"

Mat Linear::linearize(Mat inp)
{
    return inp;
}


Linear_gamma::Linear_gamma(float gamma_, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    gamma = gamma_;
}

Mat Linear_gamma::linearize(Mat inp) {
    return gamma_correction(inp, gamma);
}


Linear_color_polyfit::Linear_color_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    vector<bool> mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    Mat src_;
    Mat dst_;
    src.copyTo(src_, mask);
    cc.rgbl.copyTo(dst_, mask);
    src = src_.clone();
    dst = dst_.clone();
}


void Linear_color_polyfit::calc(void)
{
    Mat rs = src.rowRange(0, 1).clone();
    Mat gs = src.rowRange(1, 2).clone();
    Mat bs = src.rowRange(2, 3).clone();
    Mat rd = dst.rowRange(0, 1).clone();
    Mat gd = dst.rowRange(1, 2).clone();
    Mat bd = dst.rowRange(2, 3).clone();

    pr = polyfit(rs, rd, deg);
    pg = polyfit(gs, gd, deg);
    pb = polyfit(bs, bd, deg);
}


Mat Linear_color_polyfit::linearize(Mat inp)
{
    Mat r = inp.rowRange(0, 1).clone();
    Mat g = inp.rowRange(1, 2).clone();
    Mat b = inp.rowRange(2, 3).clone();

    Mat prr = poly1d(r, pr, deg);
    Mat pgg = poly1d(g, pg, deg);
    Mat pbb = poly1d(b, pb, deg);
    Mat res;
    res.push_back(prr);
    res.push_back(pgg);
    res.push_back(pbb);
    return res;
}


Linear_color_logpolyfit::Linear_color_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    vector<bool> mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    Mat src_;
    Mat dst_;
    src.copyTo(src_, mask);
    cc.rgbl.copyTo(dst_, mask);
    src = src_.clone();
    dst = dst_.clone();
}


void Linear_color_logpolyfit::calc(void)
{
    Mat rs = src.rowRange(0, 1).clone();
    Mat gs = src.rowRange(1, 2).clone();
    Mat bs = src.rowRange(2, 3).clone();
    Mat rd = dst.rowRange(0, 1).clone();
    Mat gd = dst.rowRange(1, 2).clone();
    Mat bd = dst.rowRange(2, 3).clone();

    pr = _polyfit(rs, rd, deg);
    pg = _polyfit(gs, gd, deg);
    pb = _polyfit(bs, bd, deg);
}


Mat Linear_color_logpolyfit::linearize(Mat inp)
{
    Mat r = inp.rowRange(0, 1).clone();
    Mat g = inp.rowRange(1, 2).clone();
    Mat b = inp.rowRange(2, 3).clone();
    Mat res;
    res.push_back(_lin(pr, r));
    res.push_back(_lin(pg, g));
    res.push_back(_lin(pb, b));
    return res;
    
}


Linear_gray_polyfit::Linear_gray_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    vector<bool> mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    Mat src_;
    Mat dst_;
    src.copyTo(src_, mask);
    cc.grayl.copyTo(dst_, mask);
    src = rgb2gray(src_.clone());
    dst = dst_.clone();
}


void Linear_gray_polyfit::calc(void)
{
    Mat p = polyfit(src, dst, deg);
}


Mat Linear_gray_polyfit::linearize(Mat inp)
{
    return poly1d(inp, p, deg);
}


Linear_gray_logpolyfit::Linear_gray_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {
    vector<bool> mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    Mat src_;
    Mat dst_;
    src.copyTo(src_, mask);
    cc.grayl.copyTo(dst_, mask);
    src = rgb2gray(src_.clone());
    dst = dst_.clone();
}


void Linear_gray_logpolyfit::calc(void)
{
    Mat p = _polyfit(src, dst, deg);
}


Mat Linear_gray_logpolyfit::linearize(Mat inp)
{
    return _lin(p, inp);
}


Mat Linear::_polyfit(Mat src, Mat dst, int deg) {
    Mat mask_ = (src > 0) & (dst > 0);
    Mat src_, dst_;
    src.copyTo(src_, mask_);
    dst.copyTo(dst_, mask_);
    Mat s, d;
    log(src_, s);
    log(dst_, d);
    Mat res = polyfit(s, d, deg);
    return res;
}


Mat Linear::_lin(Mat p, Mat x, int deg) {
    Mat mask_ = x <= 0;
    Mat y;
    log(x, y);
    y = poly1d(y, p, deg);
    Mat y_;
    exp(y, y_);
    Mat res;
    y_.copyTo(res, mask_);
    return res;
}


Mat Linear::polyfit(Mat src_x, Mat src_y, int order) {
    int npoints = src_x.checkVector(1);
    int nypoints = src_y.checkVector(1);
    Mat_<double> srcX(src_x), srcY(src_y);
    Mat_<double> A = Mat_<double>::ones(npoints, order + 1);
    for (int y = 0; y < npoints; ++y)
    {
        for (int x = 1; x < A.cols; ++x)
        {
            A.at<double>(y, x) = srcX.at<double>(y) * A.at<double>(y, x - 1);
        }
    }
    Mat w;
    cv::solve(A, srcY, w);
    return w;
}


Mat Linear::poly1d(Mat src, Mat w, int deg) {

    for (int x = 1; x < src.cols; ++x) {
        double res = 0;
        for (int d = deg; d > 0; d--) {
            res += pow(src.at<double>(x), d) * w.at<double>(deg - d);
        }
        src.at<double>(x) = res;
    }
    return src;
}

