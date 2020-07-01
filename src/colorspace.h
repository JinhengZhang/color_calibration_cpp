#ifndef COLORSPACE_H
#define COLORSPACE_H

#include "IO.h"

using namespace std;
using namespace cv;

class RGB_Base
{
public:
    double xr;
    double yr;
    double xg;
    double yg;
    double xb;
    double yb;
    IO io_base;
    float gamma;
    Mat _M_RGBL2XYZ_base;
    map<IO, Mat*> _M_RGBL2XYZ;
    IO _default_io;

    RGB_Base();

    Mat cal_M_RGBL2XYZ_base();
    Mat M_RGBL2XYZ_base();
    IO choose_io(IO io);
    void set_default(IO io);
    Mat M_RGBL2XYZ(IO io, bool rev);
    Mat rgbl2xyz(Mat rgbl, IO io);
    Mat xyz2rgbl(Mat xyz, IO io);
    Mat rgb2rgbl(Mat rgb);
    Mat rgbl2rgb(Mat rgbl);
    Mat rgb2xyz(Mat rgb, IO io);
    Mat xyz2rgb(Mat xyz, IO io);
    Mat rgbl2lab(Mat rgbl, IO io);
    Mat rgb2lab(Mat rgb, IO io);
};

RGB_Base::RGB_Base(void) {
    xr = 0.6400;
    yr = 0.3300;
    xg = 0.21;
    yg = 0.71;
    xb = 0.1500;
    yb = 0.0600;
    io_base = IO("D65", 2);
    gamma = 2.2;
    _M_RGBL2XYZ_base = NULL;
    _M_RGBL2XYZ = {};
    _default_io = IO("D65", 2);
}

#endif
