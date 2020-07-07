#include "ccm.h"


CCM_3x3::CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, string ccm_shape, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, float deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method, double xtol_, double ftol_)
{
    src = src_;
    IO dist_io = IO(dist_illuminant, dist_observer);
    cs = get_colorspace(colorspace);
    cs->set_default(dist_io);
    ColorChecker cc_;

    if (!dst.empty()) {
        cc_ = ColorChecker(dst, dst_colorspace, IO(dst_illuminant, dst_observer), dst_whites);
    }
    else if(colorchecker == "Macbeth_D65_2") {
        cc_ = ColorChecker(ColorChecker2005_LAB_D65_2, 'LAB', IO("D65", 2), Arange_18_24);
    }
    else if (colorchecker == "Macbeth_D50_2") {
        cc_ = ColorChecker(ColorChecker2005_LAB_D50_2, 'LAB', IO("D65", 2), Arange_18_24);
    }
    cc = ColorCheckerMetric(cc_, colorspace, dist_io);

    linear = get_linear(linear_)(gamma, deg, src, cc, saturated_threshold);

    Mat weights;
    if (!weights_list.empty()) {
        weights = weights_list;
    }
    else if (weights_coeff != 0) {
        Mat cc_lab_0 = cc.lab.rowRange(0, 1).clone();
        Mat weights_;
        pow(cc_lab_0, weights_coeff, weights_);
        weights = weights_;
    }

    Mat weight_mask = Mat::ones(1, src.rows, CV_8UC1);
    if (weights_color)
    {
        weight_mask = cc.color_mask;
    }

    vector<bool> saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    mask = saturate_mask & weight_mask;
    src_rgbl = linear->linearize(src);
    src.copyTo(src_rgb_masked, mask);
    src_rgbl.copyTo(src_rgbl_masked, mask);
    cc.rgb.copyTo(dst_rgb_masked, mask);
    cc.rgbl.copyTo(dst_rgbl_masked, mask);
    cc.lab.copyTo(dst_lab_masked, mask);

    if (!weights.data)
    {
        weights.copyTo(weights_masked, mask);
        weights_masked_norm = weights_masked / mean(weights_masked);
    }
    masked_len = src_rgb_masked.rows;
    xtol = xtol_;
    ftol = ftol_;
    ccm = NULL;
    if (distance == "rgb")
    {
        calculate_rgb();
    }
    else if(distance == "rgbl")
    {
        calculate_rgbl();
    }
    else
    {
        calculate();
    }

    if (initial_method == "least_square") {
        ccm0 = initial_white_balance(src_rgbl_masked, dst_rgbl_masked);
    }
    else if (initial_method == "least_square") {
        ccm0 = initial_least_square(src_rgbl_masked, dst_rgbl_masked);
    }
}


Mat CCM_3x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    Scalar rs = sum(src_rgbl.rowRange(0, 1).clone());
    Scalar gs = sum(src_rgbl.rowRange(1, 2).clone());
    Scalar bs = sum(src_rgbl.rowRange(2, 3).clone());
    Scalar rd = sum(dst_rgbl.rowRange(0, 1).clone());
    Scalar gd = sum(dst_rgbl.rowRange(1, 2).clone());
    Scalar bd = sum(dst_rgbl.rowRange(2, 3).clone());
    Mat initial_white_balance_ = (Mat_<float>(3, 3) << rd / rs, 0, 0, 0, gd / gs, 0, 0, 0, bd / bs);
    return initial_white_balance_;
}


Mat CCM_3x3::initial_least_square(Mat src_rgbl, Mat dst_rgbl) {
    return src_rgbl.inv() * dst_rgbl;
}


double CCM_3x3::loss_rgb(Mat ccm) {
    ccm = ccm.reshape(0, 3);
    Mat lab_est = cs->rgbl2rgb(src_rgbl_masked * ccm);
    dist = distance_s(lab_est, dst_rgb_masked, distance);
    Mat dist_;
    pow(dist, 2.0, dist_);
    if (weights.data)
    {
        dist = weights_masked_norm * dist_;
    }
    Scalar ss = sum(dist);
    return ss[0];
}


void CCM_3x3::calculate_rgb(void) {
    
    ccm0 = ccm0.reshape(0, 1);
    //res = fmin(loss_rgb, ccm0, xtol = xtol, ftol = ftol);
    if (res.data)
    {
        ccm = res.reshape(0, 3);
        double error = pow((loss_rgb(res) / masked_len), 0.5);
        cout << "ccm" << ccm << endl;
        cout << "error:" << error << endl;
    }
}


double CCM_3x3::loss_rgbl(Mat ccm) {
    Mat dist_;
    pow((dst_rgbl_masked - src_rgbl_masked * ccm), 2, dist_);
    if (weights.data)
    {
        dist = weights_masked_norm * dist_;
    }
    Scalar ss = sum(dist);
    return ss[0];
}


void CCM_3x3::calculate_rgbl(void) {
    if (weights.data)
    {
        ccm = initial_least_square(src_rgbl_masked, dst_rgbl_masked);
    }
    else
    {
        Mat w_, w;
        pow(weights_masked_norm, 0.5, w_);
        w = diag(w_);
        ccm = initial_least_square(src_rgbl_masked * w, dst_rgbl_masked * w);
    }
    double error = pow((loss_rgbl(ccm) / masked_len), 0.5);
}


double CCM_3x3::loss(Mat ccm) {
    ccm = ccm.reshape(0, 3);
    IO io_;
    Mat lab_est = cs->rgbl2lab(src_rgbl_masked * ccm, io_);
    dist = distance_s(lab_est, dst_rgb_masked, distance);
    Mat dist_;
    pow(dist, 2, dist_);
    if (weights.data)
    {
        dist = weights_masked_norm * dist;
    }
    Scalar ss = sum(dist);
    return ss[0];
}


void CCM_3x3::calculate(void) {
    ccm0 = ccm0.reshape(0, 1);
    //res = fmin(loss_rgb, ccm0, xtol = xtol, ftol = ftol);
    if (res.data)
    {
        ccm = res.reshape(0, 3);
        double error = pow((loss(res) / masked_len), 0.5);
        cout << "ccm" << ccm << endl;
        cout << "error:" << error << endl;
    }
}


void CCM_3x3::value(int number) {
    RNG rng;
    Mat_<float>rand(number, 3);
    rng.fill(rand, RNG::UNIFORM, 0, 1);
    vector<bool> mask_ = saturate(infer(rand), 0, 1);
    Scalar ss = sum(dist);
    double sat = ss[0] / number;
    cout << "sat:" << sat << endl;
    Mat rgbl = cs->rgb2rgbl(rand);
    mask_ = saturate(rgbl * ccm.inv(), 0, 1);
    dist = sum(mask_) / number;
    cout << "dist:" << dist << endl;
}


Mat CCM_3x3::infer(Mat img, bool L=false) {
    if (!ccm.data)
    {
        throw "No CCM values!";
    }
    Mat img_lin = linear->linearize(img);
    Mat img_ccm = img_lin * ccm;
    if (L)
    {
        return img_ccm;
    }
    return cs->rgbl2rgb(img_ccm);
}


Mat CCM_3x3::infer_image(string imgfile, bool L=false, int inp_size=255, int out_size=255, string out_dtype) {
    Mat img = imread(imgfile);
    Mat img_;
    cvtColor(img, img_, COLOR_BGR2RGB);
    img_= img_ / inp_size;
    Mat out = infer(img_, L);
    Mat out_ = out * out_size;
    out_.convertTo(out_, CV_8UC1, 100, 0.5);
    Mat img_out = min(max(out_, 0), out_size);
    //img_out = img_out.astype(out_dtype);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    return out_img;
}


void CCM_4x3::prepare(void) {
    src_rgbl_masked = add_column(src_rgbl_masked);
}


Mat CCM_4x3::add_column(Mat arr) {
    //return np.c_[arr, np.ones((*arr.shape[:-1], 1))];
}


Mat CCM_4x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    Scalar rs = sum(src_rgbl.rowRange(0, 1).clone());
    Scalar gs = sum(src_rgbl.rowRange(1, 2).clone());
    Scalar bs = sum(src_rgbl.rowRange(2, 3).clone());
    Scalar rd = sum(src_rgbl.rowRange(0, 1).clone());
    Scalar gd = sum(src_rgbl.rowRange(1, 2).clone());
    Scalar bd = sum(src_rgbl.rowRange(2, 3).clone());
    Mat initial_white_balance_ = (Mat_<float>(3, 3) << rd / rs, 0, 0, 0, gd / gs, 0, 0, 0, bd / bs);
    return initial_white_balance_;
}


Mat CCM_4x3::infer(Mat img, bool L=false) {
    if (!ccm.data) {
        throw "No CCM values!";
    }
    Mat img_lin = linear->linearize(img);
    Mat img_ccm = add_column(img_lin) * ccm;
    if (L) {
        return img_ccm;
    }
    return cs->rgbl2rgb(img_ccm);
}


void CCM_4x3::value(int number) {
    RNG rng;
    Mat_<float>rand(number, 3);
    rng.fill(rand, RNG::UNIFORM, 0, 1);
    vector<bool> mask_ = saturate(infer(rand), 0, 1);
    double sat = sum(mask_) / number;
    cout << "sat:" << sat << endl;
    Mat rgbl = cs->rgb2rgbl(rand);
    //up, down = self.ccm[:3, : ], self.ccm[3:, : ]
    //mask_ = saturate((rgbl - np.ones((number, 1))@down)@np.linalg.inv(up), 0, 1)
    dist = sum(mask_) / number;
    cout << "dist:" << dist << endl;
}
