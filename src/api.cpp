#include<iostream>
#include "ccm.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv;


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
} color_ca = { , 'sRGB', '', '', NULL, "Macbeth_D65_2", '3x3', {0.02, 0.98},  "sRGB", "gamma", 2.2, 3, "de00", "D65", '2', NULL, 0, false, "least_square", 1e-4, 1e-4 };

struct color_c* pcolor_c = &color_ca;

if(ccm_shape == "3x3")
{
	CCM_3x3 ccm(Mat src, struct color_c* pcolor_c);
}
else if(ccm_shape == "4x3")
{
	CCM_4x3 ccm(Mat src, struct color_c* pcolor_c);
}
