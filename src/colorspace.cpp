#include "colorspace.h"

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

Mat RGB_Base::cal_M_RGBL2XYZ_base() {
    Mat XYZr = Mat(xyY2XYZ(xr, yr)), XYZg = Mat(xyY2XYZ(xg, yg)), XYZb = Mat(xyY2XYZ(xb, yb));
    map <IO, vector<double>> illuminants = get_illuminant();
    Mat XYZw = Mat(illuminants[io_base]);
    Mat XYZ_rgbl;
    XYZ_rgbl.push_back(XYZr);
    XYZ_rgbl.push_back(XYZg);
    XYZ_rgbl.push_back(XYZb);
    XYZ_rgbl = XYZ_rgbl.t();
    Mat S = XYZ_rgbl.inv() * XYZw;
    Mat Sr = S.rowRange(0, 1).clone();
    Mat Sg = S.rowRange(1, 2).clone();
    Mat Sb = S.rowRange(2, 3).clone();
    _M_RGBL2XYZ_base.push_back(Sr * XYZr);
    _M_RGBL2XYZ_base.push_back(Sg * XYZg);
    _M_RGBL2XYZ_base.push_back(Sb * XYZb);
    _M_RGBL2XYZ_base.t();
    return _M_RGBL2XYZ_base;
}


Mat RGB_Base::M_RGBL2XYZ_base() {
    if (_M_RGBL2XYZ_base.empty()) {
        return _M_RGBL2XYZ_base;
    }
    return cal_M_RGBL2XYZ_base();
}

IO RGB_Base::choose_io(IO io) {
    if (io.m_illuminant.length() != 0) {
        return io;
    }
    return _default_io;
}

void RGB_Base::set_default(IO io) {
    _default_io = io;
}

Mat RGB_Base::M_RGBL2XYZ(IO io, bool rev = false) {
    io = choose_io(io);
    if (_M_RGBL2XYZ[io].begin() == _M_RGBL2XYZ[io].end()) {
        return _M_RGBL2XYZ[io][rev ? 1 : 0];
    }
    if (io.m_illuminant < io_base.m_illuminant || io.m_illuminant == io_base.m_illuminant) {
        _M_RGBL2XYZ[io] = { M_RGBL2XYZ_base(), M_RGBL2XYZ_base().inv() };
        return _M_RGBL2XYZ[io][rev ? 1 : 0];
    }
    Mat M_RGBL2XYZ = cam(io_base, io) * M_RGBL2XYZ_base();
    _M_RGBL2XYZ[io] = { M_RGBL2XYZ, M_RGBL2XYZ.inv() };
    return _M_RGBL2XYZ[io][rev ? 1 : 0];
}

Mat RGB_Base::rgbl2xyz(Mat rgbl, IO io) {
    io = choose_io(io);
    return rgbl * (M_RGBL2XYZ(io).t());
}

Mat RGB_Base::xyz2rgbl(Mat xyz, IO io) {
    io = choose_io(io);
    return xyz * (M_RGBL2XYZ(io, true).t());
}

Mat RGB_Base::rgb2rgbl(Mat rgb) {
    return gamma_correction(rgb, gamma);
}

Mat RGB_Base::rgbl2rgb(Mat rgbl) {
    return gamma_correction(rgbl, 1 / gamma);
}

Mat RGB_Base::rgb2xyz(Mat rgb, IO io) {
    io = choose_io(io);
    return rgbl2xyz(rgb2rgbl(rgb), io);
}

Mat RGB_Base::xyz2rgb(Mat xyz, IO io) {
    io = choose_io(io);
    return rgbl2rgb(xyz2rgbl(xyz, io));
}

Mat RGB_Base::rgbl2lab(Mat rgbl, IO io) {
    io = choose_io(io);
    return xyz2lab(rgbl2xyz(rgbl, io), io);
}

Mat RGB_Base::rgb2lab(Mat rgb, IO io) {
    io = choose_io(io);
    return rgbl2lab(rgb2rgbl(rgb), io);
}

sRGB_Base::sRGB_Base(void) {
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

float sRGB_Base::K0() {
    if (_K0) {
        return _K0;
    }
    return beta * phi;
}


float  sRGB_Base::_rgb2rgbl_ele(float x) {
    if (x > K0()) {
        return pow(((x + alpha - 1) / alpha), gamma);
    }

    else if (x >= -K0()) {
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

    else if (x >= -beta) {
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
