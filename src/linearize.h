#ifndef LINEARIZE_H
#define LINEARIZE_H

#include "utils.h"
#include "colorchecker.h"

using namespace std;
using namespace cv;

class Linear
{
public:
    Linear() {}
    Linear(float gamma_, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold) {}
    void calc(void) {}
    Mat linearize(Mat inp);
    void value(void) {}
};


class Linear_identity : public Linear
{
public:
    using Linear::Linear;
};


class Linear_gamma : public Linear
{
public:
    float gamma;
    Linear_gamma(float gamma_, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    Mat linearize(Mat inp);
};


class Linear_color_polyfit : public Linear
{
public:
    int deg;
    vector<bool> mask;
    Mat src;
    Mat dst;
    Mat pr, pg, pb;
    Linear_color_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    Mat linearize(Mat inp);

};


class Linear_color_logpolyfit : public Linear
{
public:
    int deg;
    bool mask;
    Mat src;
    Mat dst;
    Mat pr, pg, pb;
    Linear_color_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    Mat linearize(Mat inp);
    Mat _polyfit(Mat src, Mat dst, int deg);
    Mat _lin(Mat p, Mat x);
};


class Linear_gray_polyfit : public Linear
{
public:
    int deg;

    bool mask;
    Mat src;
    Mat dst;
    Mat p;
    Linear_gray_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    Mat linearize(Mat inp);
};


class Linear_gray_logpolyfit : public Linear
{
public:
    int deg;

    bool mask;
    Mat src;
    Mat dst;
    Mat p;
    Linear_gray_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, vector<double> saturated_threshold);
    void calc(void);
    Mat linearize(Mat inp);
    Mat _polyfit(Mat src, Mat dst, int deg);
    Mat _lin(Mat p, Mat x);
};


#endif
