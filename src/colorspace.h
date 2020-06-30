#ifndef COLORSPACE_H
#define COLORSPACE_H

class RGB_Base
{
public:
    float xr;
    float yr;
    float xg;
    float yg;
    float xb;
    float yb;
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

#endif
