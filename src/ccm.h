#ifndef CCM_H
#define CCM_H

#include "opencv2\opencv.hpp"
#include "opencv2\core\core.hpp"
#include "linearize.h"

using namespace std;
using namespace cv;

/*
    src:
        detected colors of ColorChecker patches;
        NOTICE: the color type is RGB not BGR, and the color values are in [0, 1];
        type: np.array(np.double)
    ============== split line =====================
    You can set reference colors either by dst dst_colorspace dst_illuminant dst_observer
        or by colorchecker;
    dst:
        the reference colors;
        NOTICE: the color type is some RGB color space or CIE Lab color space;
                If the color type is RGB, the format is RGB not BGR, and the color
                values are in [0, 1];
        type: np.array(np.double);
    dst_colorspace:
        the color space of the reference colors;
        NOTICE: it could be some RGB color space or CIE Lab color space;
                For the list of RGB color spaces supported, see the notes below;
        type: str;

    dst_illuminant:
        the illuminant of the reference color space;
        NOTICE: the option is for CIE Lab color space;
                if dst_colorspace is some RGB color space, the option would be ignored;
                For the list of the illumiant supported, see the notes below;
        type: str;

    dst_observer:
        the observer of the reference color space;
        NOTICE: the option is for CIE Lab color space;
                if dst_colorspace is some RGB color space, the option would be ignored;
                For the list of the observer supported, see the notes below;
        type: str;

    dst_whites:
        the indice list of gray colors of the reference colors;
        NOTICE: If it is set to None, some linearize method would not work;
        type: np.array(np.int);
    colorchecker:
        the name of the ColorChecker;
        Supported list:
            "Macbeth": Macbeth ColorChecker with 2deg D50;
            "Macbeth_D65_2": Macbeth ColorChecker with 2deg D65;
        NOTICE: you can either set the reference by variables starting with dst or this;
                The option works only if dst is set to None.
        type: str;

    ============== split line =====================
    ccm_shape:
        the shape of color correction matrix(CCM);
        Supported list:
            "3x3": 3x3 matrix;
            "4x3": 4x3 matrix;
        type: str

    saturated_threshold:
        the threshold to determine saturation;
        NOTICE: it is a tuple of [low, up];
                The colors in the closed interval [low, up] are reserved to participate
                in the calculation of the loss function and initialization parameters.
        type: tuple of [low, up];

    colorspace:
        the absolute color space that detected colors convert to;
        NOTICE: it could be some RGB color space;
                For the list of RGB color spaces supported, see the notes below;
        type: str;

    ============== split line =====================
    linear:
        the method of linearization;
        NOTICE: see Linearization.pdf for details;
        Supported list:
            "identity": no change is made;
            "gamma": gamma correction;
                     Need assign a value to gamma simultaneously;
            "color_polyfit": polynomial fitting channels respectively;
                             Need assign a value to deg simultaneously;
            "gray_polyfit": grayscale polynomial fitting;
                            Need assign a value to deg and dst_whites simultaneously;
            "color_logpolyfit": logarithmic polynomial fitting channels respectively;
                             Need assign a value to deg simultaneously;
            "gray_logpolyfit": grayscale Logarithmic polynomial fitting;
                            Need assign a value to deg and dst_whites simultaneously;
        type: str;

    gamma:
        the gamma value of gamma correction;
        NOTICE: only valid when linear is set to "gamma";
        type: double;

    deg:
        the degree of linearization polynomial;
        NOTICE: only valid when linear is set to "color_polyfit", "gray_polyfit",
                "color_logpolyfit" and "gray_logpolyfit";
        type: int;

    ============== split line =====================
    distance:
        the type of color distance;
        Supported list:
            'de00': ciede2000
            'de94': cie94
            'de76': cie76
            'cmc': CMC l:c (1984)
            'rgb': Euclidean distance of rgb color space;
            'rgbl': Euclidean distance of rgbl color space;
        type: str;

    dist_illuminant:
        the illuminant of CIE lab color space associated with distance function;
        NOTICE: only valid when distance is set to 'de00', 'de94', 'de76' and 'cmc';
                For the list of the illumiant supported, see the notes below;
        type: str;
    dist_observer:
        the observer of CIE lab color space associated with distance function;
        NOTICE: only valid when distance is set to 'de00', 'de94', 'de76' and 'cmc';
                For the list of the observer supported, see the notes below;
        type: str;

    ============== split line =====================
    There are some ways to set weights:
        1. set weights_list only;
        2. set weights_coeff only;
        3. set weights_color only;
        4. set weights_coeff and weights_color;
    see CCM.pdf for details;
    weights_list:
        the list of weight of each color;
        type: np.array(np.double)

    weights_coeff:
        the exponent number of L* component of the reference color in CIE Lab color space;
        type: double

    weights_color:
        if it is set to True, only non-gray color participate in the calculation
        of the loss function.
        NOTICE: only valid when dst_whites is assigned.
        type: double

    ============== split line =====================
    initial_method:
        the method of calculating CCM initial value;
        see CCM.pdf for details;
        Supported list:
            'least_square': least-squre method;
            'white_balance': white balance method;
        type: str;

    ============== split line =====================
    Supported list of illuminants:
        'A';
        'D50';
        'D55';
        'D65';
        'D75';
        'E';

    Supported list of observers:
        '2';
        '10';

    Supported list of RGB color spaces:
        'sRGB';
        'AdobeRGB';
        'WideGamutRGB';
        'ProPhotoRGB';
        'DCI_P3_RGB';
        'AppleRGB';
        'REC_709_RGB';
        'REC_2020_RGB';

    ============== split line =====================
    Abbr.
        src, s: source;
        dst, d: destination;
        io: illuminant & observer;
        sio, dio: source of io; destination of io;
        rgbl: linear RGB
        cs: color space;
        cc: Colorchecker;
        M: matrix
        ccm: color correction matrix;
        cam: chromatic adaption matrix;
*/

class CCM_3x3
{
public:
    int shape;
    cv::Mat src;
    ColorCheckerMetric cc;
    RGBBase* cs;
    Linear* linear;
    cv::Mat weights;
    cv::Mat mask;
    cv::Mat src_rgbl;
    cv::Mat src_rgb_masked;
    cv::Mat src_rgbl_masked;
    cv::Mat dst_rgb_masked;
    cv::Mat dst_rgbl_masked;
    cv::Mat dst_lab_masked;
    cv::Mat weights_masked;
    cv::Mat weights_masked_norm;
    int masked_len;
    string distance;
    cv::Mat dist;
    cv::Mat ccm;
    cv::Mat ccm0;

    CCM_3x3() {};
    CCM_3x3(cv::Mat src_, cv::Mat dst, string dst_illuminant, int dst_observer, cv::Mat dst_whites, 
        cv::Mat weights_list, string dst_colorspace = "sRGB", vector<double> saturated_threshold = { 0.02, 0.98 },
        string colorchecker = "Macbeth_D65_2", string linear_="LinearGamma", double gamma=2.2, 
        int deg=3, string distance_="de00", string dist_illuminant = "D65", int dist_observer=2,
        double weights_coeff=0, bool weights_color = false, string initial_method="least_square", 
        string colorspace = "sRGB", string ccm_shape="3x3");

    virtual void prepare(void) {}
    cv::Mat initial_white_balance(cv::Mat src_rgbl, cv::Mat dst_rgbl);
    cv::Mat initial_least_square(cv::Mat src_rgbl, cv::Mat dst_rgbl);
    void calculate_rgb(void);
    double loss_rgbl(void);
    void calculate_rgbl(void);
    void calculate(void);
    void value(int number);
    virtual cv::Mat infer(cv::Mat img, bool L=false);
    cv::Mat infer_image(string imgfile, bool L=false, int inp_size=255, int out_size=255);
    void calc(string initial_method, string distance_);
};

class CCM_4x3 : public CCM_3x3
{
public:
    using CCM_3x3::CCM_3x3;
    void prepare(void);
    cv::Mat add_column(cv::Mat arr);
    cv::Mat initial_white_balance(cv::Mat src_rgbl, cv::Mat dst_rgbl);
    cv::Mat infer(cv::Mat img, bool L);
    void value(int number);
};

/*
    Data is from https://www.imatest.com/wp-content/uploads/2011/11/Lab-data-Iluminate-D65-D50-spectro.xls
    see Miscellaneous.md for details.
*/
static cv::Mat ColorChecker2005_LAB_D50_2 = (cv::Mat_<Vec3d>(24, 1) <<
    Vec3d(37.986, 13.555, 14.059),
    Vec3d(65.711, 18.13, 17.81),
    Vec3d(49.927, -4.88, -21.925),
    Vec3d(43.139, -13.095, 21.905),
    Vec3d(55.112, 8.844, -25.399),
    Vec3d(70.719, -33.397, -0.199),
    Vec3d(62.661, 36.067, 57.096),
    Vec3d(40.02, 10.41, -45.964),
    Vec3d(51.124, 48.239, 16.248),
    Vec3d(30.325, 22.976, -21.587),
    Vec3d(72.532, -23.709, 57.255),
    Vec3d(71.941, 19.363, 67.857),
    Vec3d(28.778, 14.179, -50.297),
    Vec3d(55.261, -38.342, 31.37),
    Vec3d(42.101, 53.378, 28.19),
    Vec3d(81.733, 4.039, 79.819),
    Vec3d(51.935, 49.986, -14.574),
    Vec3d(51.038, -28.631, -28.638),
    Vec3d(96.539, -0.425, 1.186),
    Vec3d(81.257, -0.638, -0.335),
    Vec3d(66.766, -0.734, -0.504),
    Vec3d(50.867, -0.153, -0.27),
    Vec3d(35.656, -0.421, -1.231),
    Vec3d(20.461, -0.079, -0.973));

static cv::Mat ColorChecker2005_LAB_D65_2 = (cv::Mat_<Vec3d>(24, 1) <<
    Vec3d(37.542, 12.018, 13.33),
    Vec3d(65.2, 14.821, 17.545),
    Vec3d(50.366, -1.573, -21.431),
    Vec3d(43.125, -14.63, 22.12),
    Vec3d(55.343, 11.449, -25.289),
    Vec3d(71.36, -32.718, 1.636),
    Vec3d(61.365, 32.885, 55.155),
    Vec3d(40.712, 16.908, -45.085),
    Vec3d(49.86, 45.934, 13.876),
    Vec3d(30.15, 24.915, -22.606),
    Vec3d(72.438, -27.464, 58.469),
    Vec3d(70.916, 15.583, 66.543),
    Vec3d(29.624, 21.425, -49.031),
    Vec3d(55.643, -40.76, 33.274),
    Vec3d(40.554, 49.972, 25.46),
    Vec3d(80.982, -1.037, 80.03),
    Vec3d(51.006, 49.876, -16.93),
    Vec3d(52.121, -24.61, -26.176),
    Vec3d(96.536, -0.694, 1.354),
    Vec3d(81.274, -0.61, -0.24),
    Vec3d(66.787, -0.647, -0.429),
    Vec3d(50.872, -0.059, -0.247),
    Vec3d(35.68, -0.22, -1.205),
    Vec3d(20.475, 0.049, -0.972));

static cv::Mat Arange_18_24 = (cv::Mat_<double>(1, 6) << 18, 19, 20, 21, 22, 23);

// Macbeth ColorChecker with 2deg D50
static ColorChecker colorchecker_Macbeth = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", D50_2, Arange_18_24);

// Macbeth ColorChecker with 2deg D65
static ColorChecker colorchecker_Macbeth_D65_2 = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", D65_2, Arange_18_24);

#endif
