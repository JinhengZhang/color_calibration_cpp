#include "colorspace.h"

RGBBase::RGBBase(void) 
{
    // IO
    xr = 0.6400;
    yr = 0.3300;
    xg = 0.21;
    yg = 0.71;
    xb = 0.1500;
    yb = 0.0600;

    // linearization
    io_base = D65_2;
    gamma = 2.2;

    // to XYZ
    // M_RGBL2XYZ_base is matrix without chromatic adaptation
    // M_RGBL2XYZ is the one with
    // the _ prefix is the storage of calculated results
    _M_RGBL2XYZ = {};
    _default_io = D65_2;
}

/*
    calculation of M_RGBL2XYZ_base;
    see ColorSpace.pdf for details;
*/
cv::Mat RGBBase::cal_M_RGBL2XYZ_base() 
{
    cv::Mat XYZr, XYZg, XYZb;
    XYZr = cv::Mat(xyY2XYZ(xr, yr), true);
    XYZg = cv::Mat(xyY2XYZ(xg, yg), true);
    XYZb = cv::Mat(xyY2XYZ(xb, yb), true);
    cv::Mat XYZw = cv::Mat(illuminants[io_base], true);
    cv::Mat XYZ_rgbl;
    XYZ_rgbl.push_back(XYZr);
    XYZ_rgbl.push_back(XYZg);
    XYZ_rgbl.push_back(XYZb);
    XYZ_rgbl = XYZ_rgbl.reshape(0, 3);
    XYZ_rgbl = XYZ_rgbl.t();
    cv::Mat S = XYZ_rgbl.inv() * XYZw;
    cv::Mat Sr = S.rowRange(0, 1);
    cv::Mat Sg = S.rowRange(1, 2);
    cv::Mat Sb = S.rowRange(2, 3);
    _M_RGBL2XYZ_base.push_back(Sr * (XYZr.t()));
    _M_RGBL2XYZ_base.push_back(Sg * (XYZg.t()));
    _M_RGBL2XYZ_base.push_back(Sb * (XYZb.t()));
    _M_RGBL2XYZ_base = _M_RGBL2XYZ_base.t();
    return _M_RGBL2XYZ_base;
}

/* get M_RGBL2XYZ_base */
cv::Mat RGBBase::M_RGBL2XYZ_base() 
{
    if (_M_RGBL2XYZ_base.data) 
    {
        return _M_RGBL2XYZ_base;
    }
    return cal_M_RGBL2XYZ_base();
}

/* if io is unset, use io of this RGB color space */
IO RGBBase::choose_io(IO io) 
{
    IO test_;
    if (io == test_)
    {
        return  _default_io;
    }
    return io;
}

void RGBBase::set_default(IO io) 
{
    _default_io = io;
}

/*
    calculation of M_RGBL2XYZ;
    see ColorSpace.pdf for details;
*/
cv::Mat RGBBase::M_RGBL2XYZ(IO io, bool rev) 
{
    io = choose_io(io);
    if (_M_RGBL2XYZ.count(io) == 1) 
    {
        return _M_RGBL2XYZ[io][rev ? 1 : 0];
    }
    if (io == io_base) 
    {
        _M_RGBL2XYZ[io] = { M_RGBL2XYZ_base(), M_RGBL2XYZ_base().inv() };
        return _M_RGBL2XYZ[io][rev ? 1 : 0];
    }
    cv::Mat M_RGBL2XYZ = cam(io_base, io) * M_RGBL2XYZ_base();
    _M_RGBL2XYZ[io] = { M_RGBL2XYZ, M_RGBL2XYZ.inv() };
    return _M_RGBL2XYZ[io][rev ? 1 : 0];
}

cv::Mat RGBBase::rgbl2xyz(cv::Mat rgbl, IO io) 
{
    io = choose_io(io);
    cv::Mat _rgbl2xyz(rgbl.size(), rgbl.type());
    _rgbl2xyz = mult(rgbl, M_RGBL2XYZ(io).t());
    return _rgbl2xyz;
}

cv::Mat RGBBase::xyz2rgbl(cv::Mat xyz, IO io) 
{
    io = choose_io(io);
    return  mult(xyz, M_RGBL2XYZ(io, true).t());
}

cv::Mat RGBBase::rgb2rgbl(cv::Mat rgb) 
{
    return gammaCorrection(rgb, gamma);
}

cv::Mat RGBBase::rgbl2rgb(cv::Mat rgbl) 
{
    return gammaCorrection(rgbl, 1 / gamma);
}

cv::Mat RGBBase::rgb2xyz(cv::Mat rgb, IO io) 
{
    io = choose_io(io);
    return rgbl2xyz(rgb2rgbl(rgb), io);
}

cv::Mat RGBBase::xyz2rgb(cv::Mat xyz, IO io) 
{
    io = choose_io(io);
    return rgbl2rgb(xyz2rgbl(xyz, io));
}

cv::Mat RGBBase::rgbl2lab(cv::Mat rgbl, IO io) 
{
    io = choose_io(io);
    return xyz2lab(rgbl2xyz(rgbl, io), io);
}

cv::Mat RGBBase::rgb2lab(cv::Mat rgb, IO io) 
{
    io = choose_io(io);
    return rgbl2lab(rgb2rgbl(rgb), io);
}

double sRGBBase::K0() 
{
    if (_K0) 
    {
        return _K0;
    }
    return beta * phi;
}


double sRGBBase::_rgb2rgbl_ele(double x) 
{
    if (x > K0()) 
    {
        return pow(((x + alpha - 1) / alpha), gamma);
    }
    else if (x >= -K0()) 
    {
        return x / phi;
    }
    else 
    {
        return -(pow(((-x + alpha - 1) / alpha), gamma));
    }
}

/*
    linearization
    see ColorSpace.pdf for details;
*/
cv::Mat sRGBBase::rgb2rgbl(cv::Mat rgb) 
{
    int height = rgb.rows;
    int width = rgb.cols;
    int nc = rgb.channels();
    for (int row = 0; row < height; row++) 
    {
        for (int col = 0; col < width; col++) 
        {
            for (int nc_ = 0; nc_ < nc; nc_++)
                rgb.at<Vec3d>(row, col)[nc_] = _rgb2rgbl_ele(rgb.at<Vec3d>(row, col)[nc_]);
        }
    }
    return rgb;
}

double  sRGBBase::_rgbl2rgb_ele(double x) 
{
    if (x > beta) 
    {
        return alpha * pow(x, 1 / gamma) - (alpha - 1);
    }
    else if (x >= -beta) 
    {
        return x * phi;
    }
    else 
    {
        return -(alpha * pow(-x, 1 / gamma) - (alpha - 1));
    }
}

/*
    delinearization
    see ColorSpace.pdf for details;
*/
cv::Mat sRGBBase::rgbl2rgb(cv::Mat rgbl) 
{
    cv::Mat rgbl2rgbres(rgbl.size(), rgbl.type());
    int height = rgbl.rows;
    int width = rgbl.cols;
    int nc = rgbl.channels();
    for (int row = 0; row < height; row++) 
    {
        for (int col = 0; col < width; col++) 
        {
            for (int nc_ = 0; nc_ < nc; nc_++)
                rgbl2rgbres.at<Vec3d>(row, col)[nc_] = _rgbl2rgb_ele(rgbl.at<Vec3d>(row, col)[nc_]);
        }
    }
    return rgbl2rgbres;
}

/* get colorspace by str */
RGBBase* getColorspace(string colorspace) 
{
    RGBBase* p = new RGBBase;
    if (colorspace == "RGBBase") 
    {
        p = new RGBBase;
    }
    else if (colorspace == "sRGBBase") 
    {
        p = new sRGBBase;
    }
    else if (colorspace == "sRGB") 
    {
        p = new sRGB;
    }
    else if (colorspace == "AdobeRGB") 
    {
        p = new AdobeRGB;
    }
    else if (colorspace == "WideGamutRGB") 
    {
        p = new WideGamutRGB;
    }
    else if (colorspace == "ProPhotoRGB") 
    {
        p = new ProPhotoRGB;
    }
    else if (colorspace == "DCIP3RGB") 
    {
        p = new DCIP3RGB;
    }
    else if (colorspace == "AppleRGB") 
    {
        p = new AppleRGB;
    }
    else if (colorspace == "REC709RGB") 
    {
        p = new REC709RGB;
    }
    else if (colorspace == "REC2020RGB") 
    {
        p = new REC2020RGB;
    }
    return p;
}
