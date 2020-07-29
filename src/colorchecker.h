#ifndef COLORCHECKER_H
#define COLORCHECKER_H

#include <iostream>
#include <string>
#include "colorspace.h"

/*
	Difference between ColorChecker and ColorCheckerMetric
	The instance of ColorChecker describe the colorchecker by color values,
	color spaceand gray indice, which are stable for a colorchecker.
	The instance of ColorCheckerMetric adds the color space which is associated with
	the color distance functionand the colorchecker converts to.
*/

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
