#ifndef COLORCHECKER_H
#define COLORCHECKER_H

#include "colorspace.h"
//#include "utils.h"
#include "IO.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

#define COLORSPACE(colorspace) colorspace() 

using namespace std;
using namespace cv;

class ColorChecker
{
public:
	Mat lab;
	IO io;
	Mat rgb;
	char cs;
	Mat white_mask;
	Mat color_mask;
	ColorChecker(Mat color, char colorspace, IO io_, Mat whites);
};


class ColorCheckerMetric
{
public:
	char cc;
	char cs;
	IO io;
	Mat lab;
	Mat xyz;
	Mat rgbl;
	Mat lab;
	Mat grayl;
	Mat white_mask;
	Mat color_mask;
	ColorCheckerMetric(ColorChecker colorchecker, char colorspace, IO io);
};

#endif
