#include "ccm.h"

extern int loss_F_count = 0;

// After being called, the method produce a CCM_3x3 instance(a color correction model in fact) for inference.
CCM_3x3::CCM_3x3(cv::Mat src_, cv::Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, 
    cv::Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, 
    string linear_, double gamma, int deg, string distance_, string dist_illuminant, int dist_observer, 
    cv::Mat weights_list, double weights_coeff, bool weights_color, string initial_method, string ccm_shape)
{
    this->shape = (ccm_shape == "3x3") ? 3 : 4;

    // detected colors
    this->src = src_;

    // the absolute RGB color space that detected colors convert to
    IO dist_io = IO(dist_illuminant, dist_observer);
    this->cs = getColorspace(colorspace);
    cs->set_default(dist_io);
    ColorChecker cc_;

    // see notes of colorchecker.py for difference between ColorCheckerand and ColorCheckerMetric
    if (!dst.empty()) 
    {
        cc_ = ColorChecker(dst, dst_colorspace, IO(dst_illuminant, dst_observer), dst_whites);
    }
    else if (colorchecker == "Macbeth_D65_2") 
    {
        cc_ = ColorChecker(ColorChecker2005_LAB_D65_2, "LAB", D65_2, Arange_18_24);
    }
    else if (colorchecker == "Macbeth_D50_2") 
    {
        cc_ = ColorChecker(ColorChecker2005_LAB_D50_2, "LAB", D50_2, Arange_18_24);
    }
    this->cc = ColorCheckerMetric(cc_, colorspace, dist_io);

    // linear method
    this->linear = getLinear(linear_, gamma, deg, this->src, this->cc, saturated_threshold);

    // weights
    if (!weights_list.empty()) 
    {
        this->weights = weights_list;
    }
    else if (weights_coeff != 0) 
    {
        cv::Mat dChannels[3];
        split(this->cc.lab, dChannels);
        cv::Mat cc_lab_0 = dChannels[0];
        cv::Mat weights_;
        pow(cc_lab_0, weights_coeff, weights_);
        this->weights = weights_;
    }

    // mask
    // weight_mask selects non - gray colors if weights_color is True;
    // saturate_mask select non - saturated colors;
    cv::Mat weight_mask = cv::Mat::ones(src.rows, 1, CV_64FC1);
    if (weights_color) 
    {
        weight_mask = this->cc.color_mask;
    }
    cv::Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
    this->mask = (weight_mask) & (saturate_mask);
    this->mask.convertTo(this->mask, CV_64F);

    // prepare the data; _masked means colors having been filtered
    this->src_rgbl = this->linear->linearize(this->src);
    this->src_rgb_masked = maskCopyto(this->src, mask);
    this->src_rgbl_masked = maskCopyto(this->src_rgbl, mask);
    this->dst_rgb_masked = maskCopyto(this->cc.rgb, mask);
    this->dst_rgbl_masked = maskCopyto(this->cc.rgbl, mask);
    this->dst_lab_masked = maskCopyto(this->cc.lab, mask);

    // prepare the weights;
    if (this->weights.data) 
    {
        this->weights_masked = maskCopyto(this->weights, this->mask);
        this->weights_masked_norm = this->weights_masked / mean(this->weights_masked);
    }
    this->masked_len = this->src_rgb_masked.rows;
}

void CCM_3x3::calc(string initial_method, string distance_) 
{
    // empty for CCM_3x3, not empty for CCM_4x3
    prepare();

    if (initial_method == "white_balance") 
    {
        this->ccm0 = this->initial_white_balance(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else if (initial_method == "least_square") 
    {
        this->ccm0 = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }

    // distance function may affect the loss function and the fitting function
    this->distance = distance_;
    if (this->distance == "rgb") 
    {
        calculate_rgb();
    }
    else if (this->distance == "rgbl") 
    {
        calculate_rgbl();
    }
    else 
    {
        this->calculate();
    }
}

/*
    fitting nonlinear-optimization initial value by white balance:
    res = diag(mean(s_r)/mean(d_r), mean(s_g)/mean(d_g), mean(s_b)/mean(d_b))
    see CCM.pdf for details;
*/
cv::Mat CCM_3x3::initial_white_balance(cv::Mat src_rgbl, cv::Mat dst_rgbl) 
{
    cv::Mat sChannels[4];
    split(src_rgbl, sChannels);
    cv::Mat dChannels[3];
    split(dst_rgbl, dChannels);
    Scalar rs = sum(sChannels[0]);
    Scalar gs = sum(sChannels[1]);
    Scalar bs = sum(sChannels[2]);
    Scalar rd = sum(dChannels[0]);
    Scalar gd = sum(dChannels[1]);
    Scalar bd = sum(dChannels[2]);
    cv::Mat initial_white_balance_ = (cv::Mat_<double>(3, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0]);
    return initial_white_balance_;
}

/*
    fitting nonlinear-optimization initial value by least square:
    res = np.linalg.lstsq(src_rgbl, dst_rgbl)
    see CCM.pdf for details;
*/
cv::Mat CCM_3x3::initial_least_square(cv::Mat src_rgbl, cv::Mat dst_rgbl) 
{
    cv::Mat res;
    cv::Mat srcc = src_rgbl.reshape(1, 0);
    cv::Mat dstt = dst_rgbl.reshape(1, 0);
    cv::solve(srcc, dstt, res, DECOMP_NORMAL);
    return res;
}

/*
    loss function if distance function is rgb
    it is square - sum of color difference between src_rgbl@ccm and dst
*/
class loss_rgb_F : public cv::MinProblemSolver::Function 
{
public:
    CCM_3x3 ccm_loss;
    int loss_shape;

    loss_rgb_F(CCM_3x3 ccm3x3, int shape) 
    {
        ccm_loss = ccm3x3;
        loss_shape = shape;
    }
    int getDims() const 
    { 
        return 3 * loss_shape; 
    }
    double calc(const double* x) const 
    {
        cv::Mat ccm(loss_shape, 3, CV_64F);
        for (int i = 0; i < ccm.rows; i++) 
        {
            for (int j = 0; j < ccm.cols; j++) 
            {
                ccm.at<double>(i, j) = x[ccm.cols * i + j];
            }
        }
        loss_F_count++;
        cv::Mat res_loss(ccm_loss.src_rgbl_masked.size(), ccm_loss.src_rgbl_masked.type());
        res_loss = mult(ccm_loss.src_rgbl_masked, ccm);
        cv::Mat lab_est = ccm_loss.cs->rgbl2rgb(res_loss);
        cv::Mat dist = distance_s(lab_est, ccm_loss.dst_rgb_masked, ccm_loss.distance);
        cv::Mat dist_;
        pow(dist, 2.0, dist_);
        if (ccm_loss.weights.data) 
        {
            dist_ = ccm_loss.weights_masked_norm.mul(dist_);
        }
        Scalar ss = sum(dist_);
        return ss[0];
    }
};

/* calculate ccm if distance function is rgb */
void CCM_3x3::calculate_rgb(void) 
{
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_rgb_F(*this, this->shape));
    cv::Mat reshapeccm = this->ccm0.reshape(0, 1);
    solver->setFunction(ptr_F);
    RNG step_rng;
    cv::Mat_<double>step(reshapeccm.size());
    step_rng.fill(step, RNG::UNIFORM, -1, 1.);
    solver->setInitStep(step);
    double res = solver->minimize(reshapeccm);
    this->ccm = reshapeccm.reshape(0, this->shape);
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}

/*
    loss function of de76 de94 and de00
    it is square-sum of color difference between src_rgbl@ccm and dst
*/
double CCM_3x3::loss_rgbl(cv::Mat ccm) 
{
    cv::Mat dist_;
    cv::Mat dist_res = mult(this->src_rgbl_masked, this->ccm);
    cv::pow((this->dst_rgbl_masked - dist_res), 2, dist_);
    cv::Mat res_dist = dist_.reshape(1, 0);
    if (this->weights.data) 
    {
        dist_ = this->weights_masked_norm * dist_;
    }
    Scalar ss = sum(res_dist);
    return ss[0];
}

/* calculate ccm if distance function is rgbl */
void CCM_3x3::calculate_rgbl(void) 
{
    if (!this->weights.data) 
    {
        this->ccm = initial_least_square(this->src_rgbl_masked, this->dst_rgbl_masked);
    }
    else 
    {
        cv::Mat w_, w;
        pow(this->weights_masked_norm, 0.5, w_);
        w = cv::Mat::diag(w_);
        this->ccm = initial_least_square(mult(this->src_rgbl_masked, w), mult(this->dst_rgbl_masked, w));
    }
    double error = pow((loss_rgbl(this->ccm) / this->masked_len), 0.5);
    cout << "this->ccm" << this->ccm << endl;
    cout << "error" << error << endl;
}

/*
    loss function of de76 de94 and de00
    it is square-sum of color difference between src_rgbl@ccm and dst
*/
class loss_F : public cv::MinProblemSolver::Function 
{
public:
    CCM_3x3 ccm_loss;
    int loss_shape;
    loss_F(CCM_3x3 ccm3x3, int shape) 
    {
        ccm_loss = ccm3x3;
        loss_shape = shape;
    }
    int getDims() const 
    { 
        return loss_shape * 3; 
    }
    double calc(const double* x) const 
    {
        loss_F_count++;
        cv::Mat ccm(loss_shape, 3, CV_64F);
        for (int i = 0; i < ccm.rows; i++) 
        {
            for (int j = 0; j < ccm.cols; j++) 
            {
                ccm.at<double>(i, j) = x[ccm.cols * i + j];
            }
        }
        IO io_;
        cv::Mat res_loss(ccm_loss.src_rgbl_masked.size(), CV_64FC3);
        res_loss = mult(ccm_loss.src_rgbl_masked, ccm);
        cv::Mat lab_est(res_loss.size(), res_loss.type());
        lab_est = ccm_loss.cs->rgbl2lab(res_loss, io_);
        cv::Mat dist = distance_s(lab_est, ccm_loss.dst_lab_masked, ccm_loss.distance);

        cv::Mat dist_;
        pow(dist, 2, dist_);
        if (ccm_loss.weights.data) 
        {
            dist_ = ccm_loss.weights_masked_norm.mul(dist_);
        }
        Scalar ss = sum(dist_);
        return ss[0];
    }
};

/* calculate ccm if distance function is associated with CIE Lab color space */
void CCM_3x3::calculate(void) 
{
    cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
    cv::Ptr<MinProblemSolver::Function> ptr_F(new loss_F(*this, this->shape));
    solver->setFunction(ptr_F);
    cv::Mat reshapeccm = this->ccm0.reshape(0, 1);
    RNG step_rng;
    cv::Mat_<double>step(reshapeccm.size());
    step_rng.fill(step, RNG::UNIFORM, -1., 1.); 
    solver->setInitStep(step);
    double res = solver->minimize(reshapeccm);
    this->ccm = reshapeccm.reshape(0, this->shape);
    double error = pow((res / this->masked_len), 0.5);
    cout << "error:" << error << endl;
}

/*
    evaluate the model by residual error, overall saturation and coverage volume;
    see Algorithm.py for details;
    NOTICE: The method is not yet complete.
*/
void CCM_3x3::value(int number) 
{
    RNG rng;
    cv::Mat_<Vec3d>rand(number, 1);
    rng.fill(rand, RNG::UNIFORM, 0., 1.);
    cv::Mat mask_ = saturate(infer(rand, false), 0, 1);
    Scalar ss = sum(mask_);
    double sat = ss[0] / number;
    cout << "sat" << sat << endl;
    cv::Mat rgbl = this->cs->rgb2rgbl(rand);
    mask_ = saturate(mult(rgbl, this->ccm.inv()), 0, 1);
    cv::Mat mask_pre = mask_.reshape(1, 0);
    Scalar sss = sum(mask_pre);
    double dist_ = sss[0] / number;
    cout << "dist_" << dist_ << endl;
}

/* infer using calculate ccm */
cv::Mat CCM_3x3::infer(cv::Mat img, bool L) 
{
    if (!this->ccm.data)
    {
        throw "No CCM values!";
    }
    L = false;
    cv::Mat img_lin = this->linear->linearize(img);
    cv::Mat img_ccm(img_lin.size(), img_lin.type());
    img_ccm = mult3D(img_lin, this->ccm);
    if (L == true) 
    {
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}

/*
    infer image and output as an BGR image with uint8 type
    mainly for test or debug!
*/
cv::Mat CCM_3x3::infer_image(string imgfile, bool L, int inp_size, int out_size) 
{
    cv::Mat img = imread(imgfile);
    cv::Mat img_;
    cvtColor(img, img_, COLOR_BGR2RGB);
    img_.convertTo(img_, CV_64F);
    img_ = img_ / inp_size;
    cv::Mat out = this->infer(img_, L);
    cv::Mat out_ = out * out_size;
    out_.convertTo(out_, CV_8UC3);
    cv::Mat img_out = min(max(out_, 0), out_size);
    cv::Mat out_img;
    cvtColor(img_out, out_img, COLOR_RGB2BGR);
    return out_img;
}

/* see CCM.pdf for details */
void CCM_4x3::prepare(void) 
{
    this->src_rgbl_masked = add_column(this->src_rgbl_masked);
}

/* convert matrix A to [A, 1] */
cv::Mat CCM_4x3::add_column(cv::Mat arr) 
{
    cv::Mat arr1 = cv::Mat::ones(arr.size(), CV_64F);
    cv::Mat arr_out(arr.size(), CV_64FC4);
    cv::Mat arr_channels[3];
    split(arr, arr_channels);
    vector<cv::Mat> arrout_channel;
    arrout_channel.push_back(arr_channels[0]);
    arrout_channel.push_back(arr_channels[1]);
    arrout_channel.push_back(arr_channels[2]);
    arrout_channel.push_back(arr1);
    merge(arrout_channel, arr_out);
    return arr_out;
}

/* 
    fitting nonlinear-optimization initial value by white balance:
    see CCM.pdf for details; 
*/
cv::Mat CCM_4x3::initial_white_balance(cv::Mat src_rgbl, cv::Mat dst_rgbl) 
{
    cv::Mat schannels[3];
    cv::Mat dchannels[3];
    split(src_rgbl, schannels);
    split(dst_rgbl, dchannels);
    Scalar rs = sum(schannels[0]);
    Scalar gs = sum(schannels[1]);
    Scalar bs = sum(schannels[2]);
    Scalar rd = sum(dchannels[0]);
    Scalar gd = sum(dchannels[1]);
    Scalar bd = sum(dchannels[2]);
    cv::Mat initial_white_balance_ = (cv::Mat_<double>(4, 3) << rd[0] / rs[0], 0, 0, 0, gd[0] / gs[0], 0, 0, 0, bd[0] / bs[0], 0, 0, 0);
    return initial_white_balance_;
}

/* infer using calculate ccm */
cv::Mat CCM_4x3::infer(cv::Mat img, bool L) 
{
    if (!this->ccm.data) 
    {
        throw "No CCM values!";
    }
    cv::Mat img_lin = this->linear->linearize(img);
    cv::Mat img_ccm = mult(add_column(img_lin), this->ccm);
    if (L) 
    {
        return img_ccm;
    }
    return this->cs->rgbl2rgb(img_ccm);
}

/*
    evaluate the model by residual error, overall saturation and coverage volume;
    see Algorithm.py for details;
    NOTICE: The method is not yet complete.
*/
void CCM_4x3::value(int number) 
{
    RNG rng;
    cv::Mat_<Vec3d>rand(number, 1);
    rng.fill(rand, RNG::UNIFORM, 0, 1);
    cv::Mat mask_ = saturate(infer(rand, false), 0, 1);
    Scalar ss = sum(mask_);
    double sat = ss[0] / number;
    cout << "sat:" << sat << endl;
    cv::Mat rgbl = this->cs->rgb2rgbl(rand);
    cv::Mat up = this->ccm.rowRange(0, 3);
    cv::Mat down = this->ccm.rowRange(3, this->ccm.rows);
    cv::Mat ones_infer = cv::Mat::ones(number, 1, CV_64F) * down;
    cv::Mat ones_infer_ = ones_infer.reshape(3, 0);
    mask_ = saturate(mult(rgbl - ones_infer_, up.inv()), 0, 1);
    Scalar sss = sum(mask_);
    double dist_ = sss[0] / number;
    cout << "dist:" << dist_ << endl;
}
