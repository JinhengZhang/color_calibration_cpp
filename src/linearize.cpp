#include "linearize.h"

Mat Linear::linearize(Mat inp)
{
    return inp;
}

Linear_gamma::Linear_gamma(float gamma_, int deg, Mat src, ColorCheckerMetric cc, double* saturated_threshold)
{
    gamma = gamma_;
}

Mat Linear_gamma::linearize(Mat inp) {
    return gamma_correction(inp, gamma);
}



Linear_color_polyfit::Linear_color_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, double* saturated_threshold) {
    mask = saturate(src, saturated_threshold);
    Mat src_;
    Mat dst_;
    for (int i = 0; i < src.rows; i++)
    {
        if (mask[i] == true)
        {
            src_.push_back(src.row(i));
            dst_.push_back(cc.rgbl.row(i));
        }
    }
    src = src_.clone();
    dst = dst_.clone();

}


void Linear_color_polyfit::calc(void)
{
    Mat rs = src.rowRange(0, 1).clone();
    Mat gs = src.rowRange(1, 2).clone();
    Mat bs = src.rowRange(2, 3).clone();
    Mat rd = dst.rowRange(0, 1).clone();
    Mat gd = dst.rowRange(1, 2).clone();
    Mat bd = dst.rowRange(2, 3).clone();

    Mat pr = polyfit(rs, rd, deg);
    Mat pg = polyfit(gs, gd, deg);
    Mat pb = polyfit(bs, bd, deg);

}



Mat Linear_color_polyfit::linearize(Mat inp)
{
    Mat r = inp.rowRange(0, 1).clone();
    Mat g = inp.rowRange(1, 2).clone();
    Mat b = inp.rowRange(2, 3).clone();

    Mat prr = poly1d(r, pr, deg);
    Mat pgg = poly1d(g, pg, deg);
    Mat pbb = poly1d(b, pb, deg);
    Mat res;
    res.push_back(prr);
    res.push_back(pgg);
    res.push_back(pbb);
    return res;
}


Linear_color_logpolyfit::Linear_color_logpolyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, double* saturated_threshold) {
    mask = saturate(src, saturated_threshold);
    Mat src_;
    Mat dst_;
    for (int i = 0; i < src.rows; i++)
    {
        if (mask[i] == true)
        {
            src_.push_back(src.row(i));
            dst_.push_back(cc.rgbl.row(i));
        }
    }
    src = src_.clone();
    dst = dst_.clone();

}

void Linear_color_logpolyfit::calc(void)
{
    Mat rs = src.rowRange(0, 1).clone();
    Mat gs = src.rowRange(1, 2).clone();
    Mat bs = src.rowRange(2, 3).clone();
    Mat rd = dst.rowRange(0, 1).clone();
    Mat gd = dst.rowRange(1, 2).clone();
    Mat bd = dst.rowRange(2, 3).clone();
    /*
        def _polyfit(s, d, deg) :
            mask = (s > 0) & (d > 0)
            s = s[mask]
            d = d[mask]
            p = np.polyfit(np.log(s), np.log(d), deg)
            return np.poly1d(p)

            self.pr, self.pg, self.pb = _polyfit(rs, rd, self.deg), _polyfit(gs, gd, self.deg), _polyfit(bs, bd, self.deg)
    */
}

Mat Linear_color_logpolyfit::linearize(Mat inp)
{
    /*  def _lin(p, x) :
            mask = x > 0
            y = x.copy()
            y[mask] = np.exp(p(np.log(x[mask])))
            y[~mask] = 0
            return y
            r, g, b = inp[..., 0], inp[..., 1], inp[..., 2]
            return np.stack([_lin(self.pr, r), _lin(self.pg, g), _lin(self.pb, b)], axis = -1)
    */
}


Linear_gray_polyfit::Linear_gray_polyfit(float gamma, int deg, Mat src, ColorCheckerMetric cc, double* saturated_threshold) {
    mask = saturate(src, saturated_threshold) & cc.white_mask;
    Mat src_;
    Mat dst_;
    for (int i = 0; i < src.rows; i++)
    {
        if (mask[i] == true)
        {
            src_.push_back(src.row(i));
            dst_.push_back(cc.grayl.row(i));
        }
    }
    src = rgb2gray(src_.clone());
    dst = dst_.clone();

}

void Linear_gray_polyfit::calc(void)
{
    Mat p = polyfit(src, dst, deg);
    cout << 'p' << p << endl;

}

Mat Linear_gray_polyfit::linearize(Mat inp)
{
    return poly1d(inp, p, deg);
}


Linear_gray_logpolyfit::Linear_gray_logpolyfit(float gamma, float deg, Mat src, ColorCheckerMetric cc, float* saturated_threshold) {
    mask = saturate(src, saturated_threshold) & cc.white_mask;
    Mat src_;
    Mat dst_;
    for (int i = 0; i < src.rows; i++)
    {
        if (mask[i] == true)
        {
            src_.push_back(src.row(i));
            dst_.push_back(cc.grayl.row(i));
        }
    }
    src = rgb2gray(src_.clone());
    dst = dst_.clone();

}

void Linear_gray_logpolyfit::calc(void)
{
    Mat p = _polyfit(src, dst, deg);
}

Mat Linear_gray_logpolyfit::linearize(Mat inp)
{
    //return _lin(p, inp);
}


Mat _polyfit(Mat src, Mat dst, int deg) {
    mask = (src > 0) & (dst > 0);
    Mat src_;
    Mat dst_;
    for (int i = 0; i < src.rows; i++)
    {
        if (mask[i] == true)
        {
            src_.push_back(src.row(i));
            dst_.push_back(cc.rgbl.row(i));
        }
    }
    Mat s;
    Mat d;
    log(src_, s);
    log(dst_, d);
    Mat res = polyfit(s, d, deg);
    return res;
}

Mat _lin(Mat p, Mat x) {
    /*
    mask = x > 0;
    Mat y = x;
    Mat y_;
    for (int i = 0; i < src.rows; i++)
    {
        if (mask[i] == true)
        {
            src_.push_back(exp(p(src.row(i));
            dst_.push_back(cc.rgbl.row(i));
        }
    }
    return y;
    */
}

Mat polyfit(Mat src_x, Mat src_y, int order) {
    int npoints = src_x.checkVector(1);
    int nypoints = src_y.checkVector(1);
    Mat_<double> srcX(src_x), srcY(src_y);
    Mat_<double> A = Mat_<double>::ones(npoints, order + 1);
    for (int y = 0; y < npoints; ++y)
    {
        for (int x = 1; x < A.cols; ++x)
        {
            A.at<double>(y, x) = srcX.at<double>(y) * A.at<double>(y, x - 1);
        }
    }
    Mat w;
    cv::solve(A, srcY, w);
    return w;
}

Mat poly1d(Mat src, Mat w, int deg) {

    for (int x = 1; x < src.cols; ++x) {
        double res = 0.0;
        for (int d = deg; d > 0; d--) {
            res += pow(src.at<double>(x), d) * w.at<double>(deg - d);
        }
        src.at<double>(x) = res;
    }
    return src;
}

