#ifndef COLORSPACE_H
#define COLORSPACE_H

#include "utils.h"

/*
    base of RGB color space;
    the argument values are from AdobeRGB;
    Data from https://en.wikipedia.org/wiki/Adobe_RGB_color_space
*/
class RGBBase
{
public:
    double xr;
    double yr;
    double xg;
    double yg;
    double xb;
    double yb;
    double alpha;
    double beta;
    double phi;
    double _K0;
    IO io_base;
    double gamma;
    cv::Mat _M_RGBL2XYZ_base;
    map<IO, vector<cv::Mat>> _M_RGBL2XYZ;
    IO _default_io;

    RGBBase();

    virtual cv::Mat cal_M_RGBL2XYZ_base();
    virtual cv::Mat M_RGBL2XYZ_base();
    virtual IO choose_io(IO io);
    virtual void set_default(IO io);
    virtual cv::Mat M_RGBL2XYZ(IO io, bool rev = false);
    virtual cv::Mat rgbl2xyz(cv::Mat rgbl, IO io);
    virtual cv::Mat xyz2rgbl(cv::Mat xyz, IO io);
    virtual cv::Mat rgb2rgbl(cv::Mat rgb);
    virtual cv::Mat rgbl2rgb(cv::Mat rgbl);
    virtual cv::Mat rgb2xyz(cv::Mat rgb, IO io);
    virtual cv::Mat xyz2rgb(cv::Mat xyz, IO io);
    virtual cv::Mat rgbl2lab(cv::Mat rgbl, IO io);
    virtual cv::Mat rgb2lab(cv::Mat rgb, IO io);
};

/*
    base of sRGB-like color space;
    the argument values are from sRGB;
    data from https://en.wikipedia.org/wiki/SRGB
*/
class sRGBBase : public RGBBase
{  
public:
    sRGBBase():RGBBase()
    {
        // primaries
        xr = 0.6400;
        yr = 0.3300;
        xg = 0.3000;
        yg = 0.6000;
        xb = 0.1500;
        yb = 0.0600;
        alpha = 1.055;
        beta = 0.0031308;
        phi = 12.92;
        gamma = 2.4;
    }
    
    double K0();
    double _rgb2rgbl_ele(double x);
    cv::Mat rgb2rgbl(cv::Mat rgb);
    double _rgbl2rgb_ele(double x);
    cv::Mat rgbl2rgb(cv::Mat rgbl);
};

/* data from https ://en.wikipedia.org/wiki/SRGB */
class sRGB : public sRGBBase
{
public:
    sRGB() : sRGBBase() 
    {
        _M_RGBL2XYZ_base = (cv::Mat_<double>(3, 3) <<
            0.41239080, 0.35758434, 0.18048079,
            0.21263901, 0.71516868, 0.07219232,
            0.01933082, 0.11919478, 0.95053215);
    }
};

class AdobeRGB : public RGBBase 
{
public:
    using RGBBase::RGBBase;
};

/* data from https://en.wikipedia.org/wiki/Wide-gamut_RGB_color_space */
class WideGamutRGB : public RGBBase 
{
public:
    WideGamutRGB() : RGBBase() 
    {
        xr = 0.7347;
        yr = 0.2653;
        xg = 0.1152;
        yg = 0.8264;
        xb = 0.1566;
        yb = 0.0177;
        io_base = D65_2;
    }
};

/* data from https://en.wikipedia.org/wiki/ProPhoto_RGB_color_space */
class ProPhotoRGB : public RGBBase 
{
public:
    ProPhotoRGB() : RGBBase() 
    {
        xr = 0.734699;
        yr = 0.265301;
        xg = 0.159597;
        yg = 0.820403;
        xb = 0.036598;
        yb = 0.000105;
        io_base = D65_2;
    }
};

/* data from https://en.wikipedia.org/wiki/DCI-P3 */
class DCIP3RGB : public RGBBase 
{
public:
    DCIP3RGB() : RGBBase() 
    {
        xr = 0.680;
        yr = 0.32;
        xg = 0.265;
        yg = 0.69;
        xb = 0.15;
        yb = 0.06;
    }
};

/* data from http://www.brucelindbloom.com/index.html?WorkingSpaceInfo.html */
class AppleRGB : public RGBBase 
{
public:
    AppleRGB() : RGBBase() 
    {
        xr = 0.626;
        yr = 0.34;
        xg = 0.28;
        yg = 0.595;
        xb = 0.155;
        yb = 0.07;
        gamma = 1.8;
    }
};

/* data from https://en.wikipedia.org/wiki/Rec._709 */
class REC709RGB : public sRGBBase 
{
public:
    REC709RGB() : sRGBBase() 
    {
        xr = 0.64;
        yr = 0.33;
        xg = 0.3;
        yg = 0.6;
        xb = 0.15;
        yb = 0.06;
        alpha = 1.099;
        beta = 0.018;
        phi = 4.5;
        gamma = 1 / 0.45;
    }
};

/* data from https://en.wikipedia.org/wiki/Rec._2020 */
class REC2020RGB : public sRGBBase 
{
public:
    REC2020RGB() : sRGBBase() 
    {
        xr = 0.708;
        yr = 0.292;
        xg = 0.17;
        yg = 0.797;
        xb = 0.131;
        yb = 0.046;
        alpha = 1.09929682680944;
        beta = 0.018053968510807;
        phi = 4.5;
        gamma = 1 / 0.45;
    }
};

RGBBase* getColorspace(string colorspace);

#endif
