#include "ccm.h"

namespace cv {
    namespace ccm {
        ColorCorrectionModel::ColorCorrectionModel(cv::Mat src, Color dst, RGB_Base_ cs, std::string distance,
            LINEAR_TYPE linear, double gamma, int deg,
            std::vector<double> saturated_threshold, cv::Mat weights_list, double weights_coeff,
            std::string initial_method, double xtol, double ftol) :
            src(src), dst(dst), cs(cs), distance(distance), xtol(xtol), ftol(ftol) {
            cv::Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
            this->linear = get_linear(gamma, deg, this->src, this->dst, saturate_mask, this->cs, linear);
            _cal_weights_masks(weights_list, weights_coeff, saturate_mask);

            this->src_rgbl = this->linear->linearize(mask_copyto(this->src, mask));
            this->dst = this->dst[mask];
            dst_rgbl = this->dst.to(*(this->cs.l)).colors;

            if (initial_method == "white_balance") {
                initial_white_balance();
            }
            else if (initial_method == "least_square") {
                initial_least_square();
            }

            prepare();

            if (distance == "rgbl") {
                initial_least_square(true);
            }
            else {
                fitting();
            }
        };

        void ColorCorrectionModel::_cal_weights_masks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask) {
            if (weights_list.empty()) {
                weights = weights_list;
            }
            else if (weights_coeff) {
                pow(dst.toLuminant(dst.cs.io), weights_coeff, weights);
            }

            cv::Mat weight_mask = cv::Mat::ones(src.rows, 1, CV_64FC1);
            if (!weights.empty()) {
                weight_mask = weights > 0;
            }
            this->mask = (weight_mask) & (saturate_mask);

            if (!weights.empty()) {
                cv::Mat weights_masked = mask_copyto(this->weights, this->mask);
                weights = weights_masked / mean(weights_masked);
            }
            masked_len = sum(mask)[0];
        }

        cv::Mat ColorCorrectionModel::initial_white_balance(void) {
            cv::Mat sChannels[3];
            split(src_rgbl, sChannels);
            cv::Mat dChannels[3];
            split(dst_rgbl, dChannels);
            std::vector <double> initial_vec = { sum(dChannels[0])[0] / sum(sChannels[0])[0], 0, 0, 0,
                sum(dChannels[1])[0] / sum(sChannels[1])[0], 0, 0, 0, sum(dChannels[2])[0] / sum(sChannels[2])[0], 0, 0, 0 };
            std::vector <double> initial_vec_(initial_vec.begin(), initial_vec.begin() + shape);
            cv::Mat initial_white_balance_ = cv::Mat(initial_vec_, true).reshape(0, shape / 3);

            return initial_white_balance_;
        };

        cv::Mat ColorCorrectionModel::initial_least_square(bool fit) {
            cv::Mat A, B, w;
            if (weights.empty()) {
                A = src_rgbl;
                B = dst_rgbl;
            }
            else {
                pow(weights, 0.5, w);
                A = w.mul(src_rgbl);
                B = w.mul(dst_rgbl);
            }
            solve(A, B, ccm0, DECOMP_NORMAL);

            if (fit) {
                ccm = ccm0;
                cv::Mat residual = A * ccm - B;
                Scalar s = residual.dot(residual);
                double sum = s[0];
                error = sqrt(sum / masked_len);
            }
        };

        class LossFunction : public cv::MinProblemSolver::Function {
        public:
            ColorCorrectionModel* ccm_loss;
            LossFunction(ColorCorrectionModel* ccm) : ccm_loss(ccm) {};
            int getDims() const { return ccm_loss->shape; }
            double calc(const double* x) const {
                cv::Mat ccm(ccm_loss->shape, 1, CV_64F);
                for (int i = 0; i < ccm_loss->shape; i++) {
                    ccm.at<double>(i, 0) = x[i];
                }
                ccm.reshape(0, 3);
                cv::Mat dist = Color(ccm_loss->src_rgbl * ccm, ccm_loss->cs).diff(ccm_loss->dst, ccm_loss->distance);

                cv::Mat dist_;
                pow(dist, 2, dist_);
                if (!ccm_loss->weights.empty()) {
                    dist_ = ccm_loss->weights.mul(dist_);
                }
                Scalar ss = sum(dist_);
                return ss[0];
            }
        };

        void ColorCorrectionModel::fitting(void) {
            cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
            cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
            solver->setFunction(ptr_F);
            cv::Mat reshapeccm = ccm0.reshape(0, 1);
            cv::Mat_<double>step(reshapeccm.size());
            //此处增加step初始化值
            solver->setInitStep(step);
            //cout << "reshapeccm" << reshapeccm << endl;
            double res = solver->minimize(reshapeccm);
            //cout << "reshapeccm" << reshapeccm << endl;
            ccm = reshapeccm.reshape(0, shape);
            //cout << "loss_F_count  " << loss_F_count << endl;
            //cout<<"int loss_F_count"<< loss_F_count <<endl;
            //cout << "res" << res << endl;
            double error = pow((res / masked_len), 0.5);
            //cout << "error:" << error << endl;

        };

        cv::Mat ColorCorrectionModel::infer(cv::Mat img, bool L) {
            if (!ccm.data)
            {
                throw "No CCM values!";
            }
            L = false;
            cv::Mat img_lin = linear->linearize(img);
            cv::Mat img_ccm(img_lin.size(), img_lin.type());
            img_ccm = mult3D(prepare(img_lin), ccm);  //
            if (L == true) {
                return img_ccm;
            }
            return cs.fromL(img_ccm);
        };

        cv::Mat ColorCorrectionModel::infer_image(std::string imgfile, bool L, int inp_size, int out_size) {  //
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
        };

        cv::Mat ColorCorrectionModel_4x3::prepare(cv::Mat arr) {
            cv::Mat arr1 = cv::Mat::ones(arr.size(), CV_64F);
            cv::Mat arr_out(arr.size(), CV_64FC4);
            cv::Mat arr_channels[3];
            split(arr, arr_channels);
            std::vector<cv::Mat> arrout_channel;
            arrout_channel.push_back(arr_channels[0]);
            arrout_channel.push_back(arr_channels[1]);
            arrout_channel.push_back(arr_channels[2]);
            arrout_channel.push_back(arr1);
            merge(arrout_channel, arr_out);
            return arr_out;
        };

        ColorCorrectionModel* color_correction(CCM_TYPE ccm_type, cv::Mat src, Color dst, 
            RGB_Base_ cs, std::string distance, LINEAR_TYPE linear,
            double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list,
            double weights_coeff, std::string initial_method, double xtol, double ftol) {
            ColorCorrectionModel* p = new ColorCorrectionModel(src, dst, cs, distance, linear, gamma, deg,
                saturated_threshold, weights_list, weights_coeff, initial_method, xtol, ftol);
            switch (ccm_type)
            {
            case cv::ccm::ColorCorrectionModel_3x3:
                p = new ColorCorrectionModel_3x3(src, dst, cs, distance, linear,gamma, deg, 
                    saturated_threshold, weights_list, weights_coeff, initial_method, xtol, ftol);
                return p;
                break;
            case cv::ccm::ColorCorrectionModel_4x3:
                p = new ColorCorrectionModel_4x3(src, dst, cs, distance, linear, gamma, deg,
                    saturated_threshold, weights_list, weights_coeff, initial_method, xtol, ftol);
                return p;
                break;
            }
        }
    }
}