#ifndef COLORCHECKER_H
#define COLORCHECKER_H

#include <iostream>
#include <string>
#include "colorspace.h"

class ColorChecker
{
public:
	cv::Mat lab;
	IO io;
	cv::Mat rgb;
	RGBBase* cs;
	cv::Mat white_mask;
	cv::Mat color_mask;
	ColorChecker() {};
	ColorChecker(cv::Mat, string, IO, cv::Mat);
};

class ColorCheckerMetric
{
public:
	ColorChecker cc;
	RGBBase* cs;
	IO io;
	cv::Mat lab;
	cv::Mat xyz;
	cv::Mat rgb;
	cv::Mat rgbl;
	cv::Mat grayl;
	cv::Mat white_mask;
	cv::Mat color_mask;
	ColorCheckerMetric() {};
	ColorCheckerMetric(ColorChecker colorchecker, string colorspace, IO io_);
};

#endif
