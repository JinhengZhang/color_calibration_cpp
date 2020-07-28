#ifndef LINEARIZE_H
#define LINEARIZE_H

#include "colorchecker.h"

using namespace std;
using namespace cv;

class Linear
{
public:
    Linear() {}
    Linear(float gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {}
    virtual void calc(void) {}
    virtual cv::Mat linearize(cv::Mat inp);
    virtual void value(void) {}
    cv::Mat polyfit(cv::Mat src_x, cv::Mat src_y, int order);
    cv::Mat poly1d(cv::Mat src, cv::Mat w, int deg);
    cv::Mat _polyfit(cv::Mat src, cv::Mat dst, int deg);
    cv::Mat _lin(cv::Mat p, cv::Mat x, int deg);
};

class LinearIdentity : public Linear
{
public:
    using Linear::Linear;
};

class LinearGamma : public Linear
{
public:
    double gamma;
    LinearGamma() {};
    LinearGamma(float gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    cv::Mat linearize(cv::Mat inp);
};

class LinearColorPolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat pr, pg, pb;
    LinearColorPolyfit() {};
    LinearColorPolyfit(float gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

class LinearColorLogpolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat pr, pg, pb;
    LinearColorLogpolyfit() {};
    LinearColorLogpolyfit(float gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

class LinearGrayPolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat p;
    LinearGrayPolyfit() {};
    LinearGrayPolyfit(float gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

class LinearGrayLogpolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat p;
    LinearGrayLogpolyfit() {};
    LinearGrayLogpolyfit(float gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

Linear* getLinear(string linear, double gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
cv::Mat maskCopyto(cv::Mat src, cv::Mat mask);

#endif
