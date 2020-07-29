#ifndef LINEARIZE_H
#define LINEARIZE_H

#include "colorchecker.h"

using namespace std;
using namespace cv;

/* linearization base */
class Linear
{
public:
    Linear() {}
    Linear(double gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {}

    // calculate parameters
    virtual void calc(void) {}

    // inference
    virtual cv::Mat linearize(cv::Mat inp);

    // evaluate linearization model
    virtual void value(void) {}

    cv::Mat polyfit(cv::Mat src_x, cv::Mat src_y, int order);
    cv::Mat poly1d(cv::Mat src, cv::Mat w, int deg);
    cv::Mat _polyfit(cv::Mat src, cv::Mat dst, int deg);
    cv::Mat _lin(cv::Mat p, cv::Mat x, int deg);
};

/* make no change */
class LinearIdentity : public Linear
{
public:
    using Linear::Linear;
};

/*
    gamma correction;
    see Linearization.py for details;
*/
class LinearGamma : public Linear
{
public:
    double gamma;
    LinearGamma() {};
    LinearGamma(double gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    cv::Mat linearize(cv::Mat inp);
};

/*
    polynomial fitting channels respectively;
    see Linearization.py for details;
*/
class LinearColorPolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat pr, pg, pb;
    LinearColorPolyfit() {};
    LinearColorPolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);

    // monotonically increase is not guaranteed;
    // see Linearization.py for more details;
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

/*
    logarithmic polynomial fitting channels respectively;
    see Linearization.py for details;
*/
class LinearColorLogpolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat pr, pg, pb;
    LinearColorLogpolyfit() {};
    LinearColorLogpolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);

    // monotonically increase is not guaranteed;
    // see Linearization.py for more details;
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

/*
    grayscale polynomial fitting;
    see Linearization.py for details;
*/
class LinearGrayPolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat p;
    LinearGrayPolyfit() {};
    LinearGrayPolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

/*
    grayscale logarithmic polynomial fitting;
    see Linearization.py for details;
*/
class LinearGrayLogpolyfit : public Linear
{
public:
    int deg;
    cv::Mat mask;
    cv::Mat src;
    cv::Mat dst;
    cv::Mat p;
    LinearGrayLogpolyfit() {};
    LinearGrayLogpolyfit(double gamma, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    cv::Mat linearize(cv::Mat inp);
};

Linear* getLinear(string linear, double gamma_, int deg, cv::Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
cv::Mat maskCopyto(cv::Mat src, cv::Mat mask);

#endif
