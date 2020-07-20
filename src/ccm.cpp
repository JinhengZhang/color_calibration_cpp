#include "ccm.h"


CCM_3x3::CCM_3x3(Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, float deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method)
{
    this->src = src_;
    IO dist_io = IO(dist_illuminant, dist_observer);
    this->cs = get_colorspace(colorspace);
    this->cs->set_default(dist_io);
    ColorChecker cc_;

    if (!dst.empty()) {
        cc_ = ColorChecker(dst, dst_colorspace, IO(dst_illuminant, dst_observer), dst_whites);
    }
    else if(colorchecker == "Macbeth_D65_2") {
        cc_ = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", IO("D65", 2), Arange_18_24);
    }
    else if (colorchecker == "Macbeth_D50_2") {
        cc_ = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", IO("D50", 2), Arange_18_24);
    }
    //else //
    this->cc = ColorCheckerMetric(cc_, colorspace, dist_io);

    this->linear = get_linear(linear_);
    Linear linear(gamma, deg, src, this->cc, saturated_threshold);
    if (!weights_list.empty()) {
        this->weights = weights_list;
    }
    else if (weights_coeff != 0) {
        Mat cc_lab_0 = this->cc.lab.rowRange(0, 1);
        Mat weights_;
        pow(cc_lab_0, weights_coeff, weights_);
        this->weights = weights_;
    }
    //else//

    Mat weight_mask = Mat::ones(1, this->src.rows, CV_8UC1);
    if (weights_color) {
        weight_mask = this->cc.color_mask;
    }

    Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    this->mask.copyTo(weight_mask, saturate_mask);
    this->src_rgbl = linear.linearize(this->src);
    this->src.copyTo(this->src_rgb_masked, this->mask);
    this->src_rgbl.copyTo(this->src_rgbl_masked, this->mask);
    this->cc.rgb.copyTo(this->dst_rgb_masked, this->mask);
    this->cc.rgbl.copyTo(this->dst_rgbl_masked, this->mask);
    this->cc.lab.copyTo(this->dst_lab_masked, this->mask);

    if (!this->weights.data) {
        this->weights.copyTo(this->weights_masked, this->mask);
        this->weights_masked_norm = this->weights_masked / mean(this->weights_masked);
    }
    this->masked_len = this->src_rgb_masked.rows;

    if (initial_method == "white_balance") {
        this->ccm0 = this->initial_white_balance(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else if (initial_method == "least_square") {
        this->ccm0 = this->initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }

    this->distance = distance_;
    if (this->distance == "rgb") {
        this->calculate_rgb();
    }
    else if(this->distance == "rgbl") {
        this->calculate_rgbl();
    }
    else {
        this->calculate();
    }
    this->prepare();
    
}


Mat CCM_3x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    Scalar rs = sum(src_rgbl.rowRange(0, 1));
    Scalar gs = sum(src_rgbl.rowRange(1, 2));
    Scalar bs = sum(src_rgbl.rowRange(2, 3));
    Scalar rd = sum(dst_rgbl.rowRange(0, 1));
    Scalar gd = sum(dst_rgbl.rowRange(1, 2));
    Scalar bd = sum(dst_rgbl.rowRange(2, 3));
    Mat initial_white_balance_ = (Mat_<double>(3, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0]);
    return initial_white_balance_;
}


Mat CCM_3x3::initial_least_square(Mat src_rgbl, Mat dst_rgbl) {
    return src_rgbl.inv() * dst_rgbl;//
}


class loss_rgb_F : public cv::MinProblemSolver::Function, public CCM_3x3 {
public:
    int getDims() const { return 2; }
    double calc(const double* x) const {
        Mat ccm(3, 3, CV_32F, &x);
        Mat lab_est = this->cs->rgbl2rgb(this->src_rgbl_masked * ccm);
        Mat dist = distance_s(lab_est, this->dst_rgb_masked, this->distance);
        Mat dist_;
        pow(dist, 2.0, dist_);
        if (this->weights.data) {
            dist_ = this->weights_masked_norm * dist_;
        }
        Scalar ss = sum(dist_);
        return ss[0];
    }
};

void CCM_3x3::calculate_rgb(void) {
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_rgb_F());
    solver->setFunction(ptr_F);
    double res = solver->minimize(this->ccm0);
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}


double CCM_3x3::loss_rgbl(Mat ccm) {
    Mat dist_;
    cv::pow((this->dst_rgbl_masked - this->src_rgbl_masked * this->ccm), 2, dist_);
    if (this->weights.data) {
        dist_ = this->weights_masked_norm * dist_;
    }
    Scalar ss = sum(dist_);
    return ss[0];
}


void CCM_3x3::calculate_rgbl(void) {
    if (this->weights.data) {
        this->ccm = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else {
        Mat w_, w;
        pow(this->weights_masked_norm, 0.5, w_);
        w = Mat::diag(w_);
        this->ccm = initial_least_square(this->src_rgbl_masked * w, this->dst_rgbl_masked * w);
    }
    double error = pow((loss_rgbl(this->ccm) / this->masked_len), 0.5);
}


class loss_F : public cv::MinProblemSolver::Function ,public CCM_3x3 {
public:
    int getDims() const { return 2; }
    double calc(const double* x) const {
        Mat ccm(3, 3, CV_32F, &x);
        IO io_;
        Mat lab_est = this->cs->rgbl2lab(this->src_rgbl_masked * ccm, io_);
        Mat dist = distance_s(lab_est, this->dst_rgb_masked, this->distance);
        Mat dist_;
        pow(dist, 2, dist_);
        if (this->weights.data) {
            dist_ = this->weights_masked_norm * dist_;
        }
        Scalar ss = sum(dist_);
        return ss[0];
    }
};


void CCM_3x3::calculate(void) {
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_F());
    solver->setFunction(ptr_F);
    double res = solver->minimize(this->ccm0);
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}


void CCM_3x3::value(int number) {
    RNG rng;
    Mat_<double>rand(number, 3);
    rng.fill(rand, RNG::UNIFORM, 0., 1.);
    Mat mask_ = saturate(infer(rand,false), 0, 1);
    Scalar ss = sum(mask);
    double sat = ss[0] / number;
    cout << "sat:" << sat << endl;
    Mat rgbl = this->cs->rgb2rgbl(rand);
    mask_ = saturate(rgbl * this->ccm.inv(), 0, 1);
    Scalar sss = sum(mask_);
    double dist_ = sss[0] / number;
    cout << "dist:" << dist_ << endl;
}


Mat CCM_3x3::infer(Mat img, bool L=false) {
    if (!this->ccm.data)
    {
        throw "No CCM values!";
    }
    Mat img_lin = this->linear->linearize(img);
    Mat img_ccm = img_lin * this->ccm;
    if (L == true){
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}


Mat CCM_3x3::infer_image(string imgfile, bool L=false, int inp_size=255, int out_size=255) {
    Mat img = imread(imgfile);
    Mat img_;
    cvtColor(img, img_, COLOR_BGR2RGB);
    img_= img_ / inp_size;
    Mat out = infer(img_, L);
    Mat out_ = out * out_size;
    out_.convertTo(out_, CV_8UC1, 100, 0.5);
    Mat img_out = min(max(out_, 0), out_size);
    Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    return out_img;
}


void CCM_4x3::prepare(void) {
    this->src_rgbl_masked = add_column(this->src_rgbl_masked);
}


Mat CCM_4x3::add_column(Mat arr) {
    Mat arr1 = Mat::ones(arr.rows, 1, CV_8U);
    Mat arr_out;
    vconcat(arr, arr1, arr_out);
    return arr_out;
}


Mat CCM_4x3::initial_white_balance(Mat src_rgbl, Mat dst_rgbl) {
    Scalar rs = sum(src_rgbl.rowRange(0, 1));
    Scalar gs = sum(src_rgbl.rowRange(1, 2));
    Scalar bs = sum(src_rgbl.rowRange(2, 3));
    Scalar rd = sum(src_rgbl.rowRange(0, 1));
    Scalar gd = sum(src_rgbl.rowRange(1, 2));
    Scalar bd = sum(src_rgbl.rowRange(2, 3));
    Mat initial_white_balance_ = (Mat_<double>(3, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0]);//
    return initial_white_balance_;
}


Mat CCM_4x3::infer(Mat img, bool L=false) {
    if (!this->ccm.data) {
        throw "No CCM values!";
    }
    Mat img_lin = this->linear->linearize(img);
    Mat img_ccm = add_column(img_lin) * this->ccm;
    if (L) {
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}


void CCM_4x3::value(int number) {
    RNG rng;
    Mat_<double>rand(number, 3);
    rng.fill(rand, RNG::UNIFORM, 0, 1);
    Mat mask_ = saturate(infer(rand,false), 0, 1);
    Scalar ss = sum(mask_);
    double sat = ss[0] / number;
    cout << "sat:" << sat << endl;
    Mat rgbl = this->cs->rgb2rgbl(rand);
    Mat up = this->ccm.rowRange(1, 3);
    Mat down = this->ccm.rowRange(3, this->ccm.rows);
    mask_ = saturate((rgbl - Mat::ones(number, 1, CV_8U) * down) * up.inv(), 0, 1);
    Scalar sss = sum(mask_);
    double dist_ = sss[0] / number;
    cout << "dist:" << dist_ << endl;
}
