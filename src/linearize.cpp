#include "linearize.h"

cv::Mat Linear::linearize(cv::Mat inp)
{
    return inp;
}

cv::Mat maskCopyto(cv::Mat src, cv::Mat mask) 
{
    cv::Mat src_(countNonZero(mask), 1, src.type());
    int countone = 0;
    for (int i = 0; i < mask.rows; i++) 
    {
        if (mask.at<double>(i, 0)) 
        {
            for (int c = 0; c < src.channels(); c++) 
            {
                src_.at<Vec3d>(countone, 0)[c] = src.at<Vec3d>(i, 0)[c];
            }
            countone++;
        };
    }
    return src_;
}

LinearGamma::LinearGamma(double gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    gamma = gamma_;
}

cv::Mat LinearGamma::linearize(cv::Mat inp) 
{
    return gammaCorrection(inp, gamma);
}

LinearColorPolyfit::LinearColorPolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    cv::Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    cv::Mat src_(countNonZero(mask), 1, src.type());
    cv::Mat dst_(countNonZero(mask), 1, cc.rgbl.type());
    this->deg = deg;
    this->src = maskCopyto(src, mask);
    this->dst = maskCopyto(cc.rgbl, mask);
    calc();
}

void LinearColorPolyfit::calc(void)
{
    cv::Mat sChannels[3];
    cv::Mat dChannels[3];
    split(this->src, sChannels);
    split(this->dst, dChannels);
    cv::Mat rs = sChannels[0];
    cv::Mat gs = sChannels[1];
    cv::Mat bs = sChannels[2];
    cv::Mat rd = dChannels[0];
    cv::Mat gd = dChannels[1];
    cv::Mat bd = dChannels[2];
    pr = polyfit(rs, rd, deg);
    pg = polyfit(gs, gd, deg);
    pb = polyfit(bs, bd, deg);
}

cv::Mat LinearColorPolyfit::linearize(cv::Mat inp)
{
    cv::Mat inpChannels[3];
    split(inp, inpChannels);
    vector<cv::Mat> channel;
    cv::Mat res;
    channel.push_back(poly1d(inpChannels[0], pr, deg));
    channel.push_back(poly1d(inpChannels[1], pg, deg));
    channel.push_back(poly1d(inpChannels[2], pb, deg));
    merge(channel, res);
    return res;
}

LinearColorLogpolyfit::LinearColorLogpolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    cv::Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    cv::Mat src_(countNonZero(mask), 1, src.type());
    cv::Mat dst_(countNonZero(mask), 1, cc.rgbl.type());
    this->deg = deg;
    this->src = maskCopyto(src, mask);
    this->dst = maskCopyto(cc.rgbl, mask);
    calc();
}

void LinearColorLogpolyfit::calc(void)
{
    cv::Mat sChannels[3];
    cv::Mat dChannels[3];
    split(src, sChannels);
    split(dst, dChannels);
    cv::Mat rs = sChannels[0];
    cv::Mat gs = sChannels[1];
    cv::Mat bs = sChannels[2];
    cv::Mat rd = dChannels[0];
    cv::Mat gd = dChannels[1];
    cv::Mat bd = dChannels[2];
    pr = _polyfit(rs, rd, deg);
    pg = _polyfit(gs, gd, deg);
    pb = _polyfit(bs, bd, deg);
}

cv::Mat LinearColorLogpolyfit::linearize(cv::Mat inp)
{
    cv::Mat channels[3];
    split(inp, channels);
    cv::Mat r = channels[0];
    cv::Mat g = channels[1];
    cv::Mat b = channels[2];
    vector<cv::Mat> channel;
    cv::Mat res;
    channel.push_back(_lin(pr, r, deg));
    channel.push_back(_lin(pg, g, deg));
    channel.push_back(_lin(pb, b, deg));
    merge(channel, res);
    return res;
}

LinearGrayPolyfit::LinearGrayPolyfit(double gamma, int deg_, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    cv::Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]) & ~cc.white_mask;
    cv::Mat src_(countNonZero(mask), 1, src.type());
    cv::Mat dst_(countNonZero(mask), 1, cc.grayl.type());
    deg = deg_;
    cv::Mat src_gray = maskCopyto(src, mask);

    // the grayscale function is approximate for src is in relative color space;
    // see Linearization.py for more details;
    this->src = rgb2gray(src_gray);
    this->dst = maskCopyto(cc.grayl, mask);
    calc();
}

void LinearGrayPolyfit::calc(void)
{
    // monotonically increase is not guaranteed;
    // see Linearization.py for more details;
    this->p = polyfit(src, dst, deg);
}

cv::Mat LinearGrayPolyfit::linearize(cv::Mat inp)
{
    cv::Mat inpChannels[3];
    split(inp, inpChannels);
    vector<cv::Mat> channel;
    cv::Mat res;
    channel.push_back(poly1d(inpChannels[0], p, deg));
    channel.push_back(poly1d(inpChannels[1], p, deg));
    channel.push_back(poly1d(inpChannels[2], p, deg));
    merge(channel, res);
    return res;
}


LinearGrayLogpolyfit::LinearGrayLogpolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold)
{
    cv::Mat mask = saturate(src, saturated_threshold[0], saturated_threshold[1]) & ~cc.white_mask;
    this->deg = deg;
    cv::Mat src_gray = maskCopyto(src, mask);
    
    // the grayscale function is approximate for src is in relative color space;
    // see Linearization.py for more details;
    this->src = rgb2gray(src_gray);
    this->dst = maskCopyto(cc.grayl, mask);
    calc();
}

void LinearGrayLogpolyfit::calc(void)
{
    // monotonically increase is not guaranteed;
    // see Linearization.py for more details;
    this->p = _polyfit(src, dst, deg);
}

cv::Mat LinearGrayLogpolyfit::linearize(cv::Mat inp)
{
    cv::Mat inpChannels[3];
    split(inp, inpChannels);
    vector<cv::Mat> channel;
    cv::Mat res;
    channel.push_back(_lin(p, inpChannels[0], deg));
    channel.push_back(_lin(p, inpChannels[1], deg));
    channel.push_back(_lin(p, inpChannels[2], deg));
    merge(channel, res);
    return res;
}

/* values less than or equal to 0 cannot participate in calculation for features of the logarithmic function. */
cv::Mat Linear::_polyfit(cv::Mat src, cv::Mat dst, int deg) {
    // polyfit for s>0 and d>0
    cv::Mat mask_ = (src > 0) & (dst > 0);
    mask_.convertTo(mask_, CV_64F);
    cv::Mat src_, dst_;
    src_ = maskCopyto(src, mask_);
    dst_ = maskCopyto(dst, mask_);
    cv::Mat s, d;
    log(src_, s);
    log(dst_, d);
    cv::Mat res = polyfit(s, d, deg);
    return res;
}

cv::Mat Linear::_lin(cv::Mat p, cv::Mat x, int deg) 
{
    cv::Mat mask_ = x >= 0;
    cv::Mat y;
    log(x, y);
    y = poly1d(y, p, deg);
    cv::Mat y_;
    exp(y, y_);
    cv::Mat res;
    y_.copyTo(res, mask_);
    return res;
}

cv::Mat Linear::polyfit(cv::Mat src_x, cv::Mat src_y, int order) 
{
    int npoints = src_x.checkVector(1);
    int nypoints = src_y.checkVector(1);
    cv::Mat_<double> srcX(src_x), srcY(src_y);
    cv::Mat_<double> A = cv::Mat_<double>::ones(npoints, order + 1);
    for (int y = 0; y < npoints; ++y)
    {
        for (int x = 1; x < A.cols; ++x)
        {
            A.at<double>(y, x) = srcX.at<double>(y) * A.at<double>(y, x - 1);
        }
    }
    cv::Mat w;
    cv::solve(A, srcY, w, DECOMP_SVD);
    return w;
}

cv::Mat Linear::poly1d(cv::Mat src, cv::Mat w, int deg) 
{
    cv::Mat res_polyfit(src.size(), src.type());
    for (int i = 0; i < src.rows; i++) 
    {
        for (int j = 0; j < src.cols; j++) 
        {
            double res = 0;
            for (int d = 0; d <= deg; d++) 
            {
                res += pow(src.at<double>(i, j), d) * w.at<double>(d, 0);
                res_polyfit.at<double>(i, j) = res;
            }
        }
    }
    return res_polyfit;
}

/* get linear by str */
Linear* getLinear(string linear, double gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) 
{
    Linear* p = new Linear(gamma_, deg, src, cc, saturated_threshold);
    if (linear == "Linear") 
    {
        p = new Linear(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "LinearIdentity") 
    {
        p = new LinearIdentity(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "LinearGamma") 
    {
        p = new LinearGamma(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "LinearColorPolyfit") 
    {
        p = new LinearColorPolyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "LinearColorLogpolyfit") 
    {
        p = new LinearColorLogpolyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "LinearGgrayPolyfit") 
    {
        p = new LinearGrayPolyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    else if (linear == "LinearGrayLogpolyfit") 
    {
        p = new LinearGrayLogpolyfit(gamma_, deg, src, cc, saturated_threshold);
    }
    return p;
}
