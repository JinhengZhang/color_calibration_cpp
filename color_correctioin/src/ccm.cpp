#include "precomp.hpp"

namespace cv {
	namespace ccm {



		ColorCorrectionModel::ColorCorrectionModel(cv::Mat src, Color dst, RGB_Base_& cs, DISTANCE_TYPE distance,
			LINEAR_TYPE linear, double gamma, int deg, std::vector<double> saturated_threshold, cv::Mat weights_list,
			double weights_coeff, INITIAL_METHOD_TYPE initial_method_type, int maxCount, double epsilon, int shape) :
			src(src), dst(dst), cs(cs), distance(distance), maxCount(maxCount), epsilon(epsilon), shape(shape)
		{
			cv::Mat saturate_mask = saturate(src, saturated_threshold[0], saturated_threshold[1]);
			this->linear = get_linear(gamma, deg, this->src, this->dst, saturate_mask, this->cs, linear);
			_cal_weights_masks(weights_list, weights_coeff, saturate_mask);

			src_rgbl = this->linear->linearize(mask_copyto(this->src, mask));
			// this->dst = this->dst[mask];
			this->dst = this->dst[mask];
			dst_rgbl = mask_copyto(this->dst.to(*(this->cs.l)).colors, mask);
			std::cout << "src_rgbl " << src_rgbl << std::endl;
			std::cout << "dst_rgbl" << dst_rgbl << std::endl;

			// empty for CCM_3x3, not empty for CCM_4x3
			src_rgbl = prepare(src_rgbl);

			// distance function may affect the loss function and the fitting function
			switch (this->distance)
			{
			case cv::ccm::RGBL:
				initial_least_square(true);
				break;
			default:
				switch (initial_method_type)
				{
				case cv::ccm::WHITE_BALANCE:
					initial_white_balance();
					break;
				case cv::ccm::LEAST_SQUARE:
					initial_least_square();
					break;
				}
				fitting();
			}
		}

		// calculate weights and mask
		void ColorCorrectionModel::_cal_weights_masks(cv::Mat weights_list, double weights_coeff, cv::Mat saturate_mask) {
			// weights
			if (!weights_list.empty()) {
				weights = weights_list;
			}
			else if (weights_coeff != 0) {
				pow(dst.toLuminant(dst.cs.io), weights_coeff, weights);
			}

			// masks
			cv::Mat weight_mask = cv::Mat::ones(src.rows, 1, CV_8U);
			if (!weights.empty()) {
				weight_mask = weights > 0;
			}
			this->mask = (weight_mask) & (saturate_mask);

			// weights' mask
			if (!weights.empty()) {
				cv::Mat weights_masked = mask_copyto(this->weights, this->mask);
				weights = weights_masked / mean(weights_masked);
			}
			masked_len = sum(mask)[0];
		};

		// make no change for ColorCorrectionModel_3x3 class
		// convert matrix A to [A, 1] in ColorCorrectionModel_4x3 class
		cv::Mat ColorCorrectionModel::prepare(cv::Mat inp) {
			if (shape == 9) {
				return inp;
			}
			else if (shape == 12) {
				cv::Mat arr1 = cv::Mat::ones(inp.size(), CV_64F);
				cv::Mat arr_out(inp.size(), CV_64FC4);
				cv::Mat arr_channels[3];
				split(inp, arr_channels);
				merge(std::vector<Mat>{ arr_channels[0], arr_channels[1], arr_channels[2], arr1 }, arr_out);
				return arr_out;

			}
		};

		// fitting nonlinear - optimization initial value by white balance :
		// res = diag(mean(s_r) / mean(d_r), mean(s_g) / mean(d_g), mean(s_b) / mean(d_b))
		// see CCM.pdf for details;
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

		// fitting nonlinear-optimization initial value by least square:
		// res = np.linalg.lstsq(src_rgbl, dst_rgbl)
		// see CCM.pdf for details;
		// if fit==True, return optimalization for rgbl distance function;

		void ColorCorrectionModel::initial_least_square(bool fit) {
			cv::Mat A, B, w;
			if (weights.empty()) {
				A = src_rgbl;
				B = dst_rgbl;
			}
			else {
				pow(weights, 0.5, w);
				cv::Mat w_;
				merge(std::vector<Mat>{ w, w, w }, w_);
				A = w_.mul(src_rgbl);
				B = w_.mul(dst_rgbl);
			}

			solve(A.reshape(1, A.rows), B.reshape(1, B.rows), ccm0, DECOMP_SVD);
			//std::cout << "ccm0   " << ccm0 << std::endl;
			if (fit) {
				ccm = ccm0;
				cv::Mat residual = A.reshape(1, A.rows) * ccm.reshape(0, shape / 3) - B.reshape(1, B.rows);
				Scalar s = residual.dot(residual);
				double sum = s[0];
				error = sqrt(sum / masked_len);
			}
		};


		class LossFunction : public cv::MinProblemSolver::Function {
		public:
			ColorCorrectionModel* ccm_loss;
			LossFunction(ColorCorrectionModel* ccm) : ccm_loss(ccm) {};

			int getDims() const {
				//std::cout << "ccm_loss->shape" << ccm_loss->shape << std::endl;
				return ccm_loss->shape;
			}
			double calc(const double* x) const {
				cv::Mat ccm(ccm_loss->shape, 1, CV_64F);
				for (int i = 0; i < ccm_loss->shape; i++) {
					ccm.at<double>(i, 0) = x[i];
				}
				ccm = ccm.reshape(0, ccm_loss->shape / 3);
				Mat reshapecolor = ccm_loss->src_rgbl.reshape(1, 0) * ccm;
				cv::Mat dist = Color(reshapecolor.reshape(3, 0), ccm_loss->cs).diff(ccm_loss->dst, ccm_loss->distance);
				//  cv::Mat dist = Color(ccm_loss->src_rgbl * ccm, ccm_loss->cs).diff(ccm_loss->dst, ccm_loss->distance);

				cv::Mat dist_;
				pow(dist, 2, dist_);
				if (!ccm_loss->weights.empty()) {
					dist_ = ccm_loss->weights.mul(dist_);
				}
				Scalar ss = sum(dist_);
				return ss[0];
			}
		};

		// fitting ccm if distance function is associated with CIE Lab color space
		void ColorCorrectionModel::fitting(void) {
			cv::Ptr<DownhillSolver> solver = cv::DownhillSolver::create();
			cv::Ptr<LossFunction> ptr_F(new LossFunction(this));
			solver->setFunction(ptr_F);
			cv::Mat reshapeccm = ccm0.reshape(0, 1);
			std::cout << " ccm0 " << ccm0 << std::endl;
			cv::Mat step = cv::Mat::ones(reshapeccm.size(), CV_64F) * 10;
			solver->setInitStep(step);
			/* TermCriteria termcrit = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, maxCount, epsilon);
			 solver->setTermCriteria(termcrit);*/
			double res = solver->minimize(reshapeccm);
			ccm = reshapeccm.reshape(0, shape);
			double error = pow((res / masked_len), 0.5);
			std::cout << " ccm " << ccm << std::endl;
			std::cout << " error " << error << std::endl;
		};

		cv::Mat ColorCorrectionModel::infer(cv::Mat img, bool L) {
			if (!ccm.data)
			{
				throw "No CCM values!";
			}
			L = false;
			cv::Mat img_lin = linear->linearize(img);
			cv::Mat img_ccm(img_lin.size(), img_lin.type());
			cv::Mat ccm_ = ccm.reshape(0, shape / 3);
			img_ccm = multiple(prepare(img_lin), ccm_);
			if (L == true) {
				return img_ccm;
			}
			return cs.fromL(img_ccm);
		};

		// infer image and output as an BGR image with uint8 type
		// mainly for test or debug!
		cv::Mat ColorCorrectionModel::infer_image(std::string imgfile, bool L, int inp_size, int out_size) {
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
	}

}


