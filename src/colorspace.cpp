#include<iostream>
#include<cmath>
//#include "utils.h"
#include "IO.h"
#include "colorspace.h"
#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"

using namespace std;
using namespace cv;

RGB_Base::RGB_Base() {
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

Mat RGB_Base::cal_M_RGBL2XYZ_base() {
    vector<double> XYZr = xyY2XYZ(xr, yr);
    vector<double> XYZg = xyY2XYZ(xg, yg);
    vector<double> XYZb = xyY2XYZ(xb, yb);
    map <IO, vector<double>> illuminants = get_illuminant();
    vector<double> XYZw = illuminants[io_base];
    Mat XYZ_rgbl;
    XYZ_rgbl.push_back(XYZr);
    XYZ_rgbl.push_back(XYZg);
    XYZ_rgbl.push_back(XYZb);
    XYZ_rgbl = XYZ_rgbl.t();
    Mat S = XYZ_rgbl.inv() * XYZw;
    double Sr = S[0];
    double Sg = S[1];
    double Sb = S[2];
    _M_RGBL2XYZ_base.push_back(Sr * XYZr);
    _M_RGBL2XYZ_base.push_back(Sg * XYZg);
    _M_RGBL2XYZ_base.push_back(Sb * XYZb);
    _M_RGBL2XYZ_base = _M_RGBL2XYZ_base.t();
    return _M_RGBL2XYZ_base;
}

Mat RGB_Base::M_RGBL2XYZ_base() {
    if (_M_RGBL2XYZ_base) {
        return _M_RGBL2XYZ_base;
    }
    return cal_M_RGBL2XYZ_base();
}

IO RGB_Base::choose_io(IO io = NULL) {
    return io or _default_io;
}
void RGB_Base::set_default(IO io) {
    _default_io = io;
}

Mat RGB_Base::M_RGBL2XYZ(IO io = NULL, bool rev = false) {
    io = choose_io(io);
    if (io in _M_RGBL2XYZ) {
        return _M_RGBL2XYZ[io][rev ? 1 : 0];
    }
    if (io == io_base) {
        _M_RGBL2XYZ[io] = (M_RGBL2XYZ_base(), M_RGBL2XYZ_base().inv());
        return _M_RGBL2XYZ[io][rev ? 1 : 0];
    }
    M_RGBL2XYZ = cam(io_base, io) * M_RGBL2XYZ_base();
    _M_RGBL2XYZ[io] = (M_RGBL2XYZ, M_RGBL2XYZ.inv());
    return _M_RGBL2XYZ[io][rev ? 1 : 0];
}

Mat RGB_Base::rgbl2xyz(Mat rgbl, IO io = NULL) {
    io = choose_io(io);
    return rgbl * (M_RGBL2XYZ(io).t());
}

Mat RGB_Base::xyz2rgbl(Mat xyz, IO io = NULL) {
    io = choose_io(io);
    return xyz * (M_RGBL2XYZ(io, true).t());
}

Mat RGB_Base::rgb2rgbl(Mat rgb) {
    return gamma_correction(rgb, gamma);
}

Mat RGB_Base::rgbl2rgb(Mat rgbl) {
    return gamma_correction(rgbl, 1 / gamma);
}

Mat RGB_Base::rgb2xyz(Mat rgb, IO io = NULL) {
    io = choose_io(io);
    return rgbl2xyz(rgb2rgbl(rgb), io);
}

Mat RGB_Base::xyz2rgb(Mat xyz, IO io = NULL) {
    io = choose_io(io);
    return rgbl2rgb(xyz2rgbl(xyz, io));
}

Mat RGB_Base::rgbl2lab(Mat rgbl, IO io = NULL) {
    io = choose_io(io);
    return xyz2lab(rgbl2xyz(rgbl, io), io);
}

Mat RGB_Base::rgb2lab(Mat rgb, IO io = NULL) {
    io = choose_io(io);
    return rgbl2lab(rgb2rgbl(rgb), io);
}


class sRGB_Base : public RGB_Base
{
public:
    float xr;
    float yr;
    float xg;
    float yg;
    float xb;
    float yb;
    float alpha;
    float beta;
    float phi;
    float gamma;
    float _K0;

    sRGB_Base();

    float K0();
    float _rgb2rgbl_ele(float x);
    Mat rgb2rgbl(Mat rgb);
    float _rgbl2rgb_ele(float x);
    Mat rgbl2rgb(Mat rgbl);

};


float sRGB_Base::K0() {
    if (_K0) {
        return _K0;
    }
    return beta * phi;
}


float  sRGB_Base::_rgb2rgbl_ele(float x) {
    if (x > K0) {
        return pow(((x + alpha - 1) / alpha), gamma);
    }

    else if (x >= -K0) {
        return x / phi;
    }

    else {
        return -(pow(((-x + alpha - 1) / alpha), gamma));
    }

}
Mat  sRGB_Base::rgb2rgbl(Mat rgb) {
    int height = rgb.rows;
    int width = rgb.cols;
    int nc = rgb.channels();
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (nc == 1) {
                rgb.at<float>(row, col) = _rgb2rgbl_ele(rgb.at<float>(row, col));
            }
            else if (nc == 3) {
                for (int nc_ = 0; nc_ < nc; nc_++)
                    rgb.at<Vec3b>(row, col)[nc_] = _rgb2rgbl_ele(rgb.at<Vec3b>(row, col)[nc_]);
            }
        }
    }
    return rgb;
}

float  sRGB_Base::_rgbl2rgb_ele(float x) {
    if (x > beta) {
        return pow(((x + alpha - 1) / alpha), gamma);
    }

    else if (x >= -bate) {
        return x * phi;
    }

    else {
        return -(pow(((-x + alpha - 1) / alpha), gamma));
    }

}

Mat  sRGB_Base::rgbl2rgb(Mat rgbl) {
    int height = rgbl.rows;
    int width = rgbl.cols;
    int nc = rgbl.channels();
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            if (nc == 1) {
                rgbl.at<float>(row, col) = _rgbl2rgb_ele(rgbl.at<float>(row, col));
            }
            else if (nc == 3) {
                for (int nc_ = 0; nc_ < nc; nc_++)
                    rgbl.at<Vec3b>(row, col)[nc_] = _rgbl2rgb_ele(rgbl.at<Vec3b>(row, col)[nc_]);
            }
        }
    }
    return rgbl;
}


class sRGB : public sRGB_Base
{
    Mat _M_RGBL2XYZ_base;
    sRGB() : sRGB_Base() {
        Mat _M_RGBL2XYZ_base = (Mat_<double>(3, 3) <<
            0.41239080, 0.35758434, 0.18048079,
            0.21263901, 0.71516868, 0.07219232,
            0.01933082, 0.11919478, 0.95053215);
    }
};

class AdobeRGB : public RGB_Base {

};

class WideGamutRGB : public RGB_Base {
    WideGamutRGB() : RGB_Base() {
        xr = 0.7347;
        yr = 0.2653;
        xg = 0.1152;
        yg = 0.8264;
        xb = 0.1566;
        yb = 0.0177;
        io_base = D50_2;
    }
};

class ProPhotoRGB : public RGB_Base {
    ProPhotoRGB() : RGB_Base() {
        xr = 0.734699;
        yr = 0.265301;
        xg = 0.159597;
        yg = 0.820403;
        xb = 0.036598;
        yb = 0.000105;
        io_base = D50_2;
    }
};

class DCI_P3_RGB : public RGB_Base {
    DCI_P3_RGB() : RGB_Base() {
        xr = 0.680;
        yr = 0.32;
        xg = 0.265;
        yg = 0.69;
        xb = 0.15;
        yb = 0.06;
    }
};

class AppleRGB : public RGB_Base {
    AppleRGB() : RGB_Base() {
        xr = 0.626;
        yr = 0.34;
        xg = 0.28;
        yg = 0.595;
        xb = 0.155;
        yb = 0.07;
        gamma = 1.8;
    }
};

class REC_709_RGB : public sRGB_Base {
    REC_709_RGB() : sRGB_Base() {
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

class REC_2020_RGB : public sRGB_Base {
    REC_2020_RGB() : sRGB_Base() {
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


Mat colorconvert(Mat color, Mat src, Mat dst) {
    return dst.xyz2rgb(src.rgb2xyz(color, D65_2), D65_2);
}
