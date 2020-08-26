#ifndef LINEARIZE_H
#define LINEARIZE_H

#include "opencv2/ccm/color.hpp"

namespace cv {
    namespace ccm {
        enum LINEAR_TYPE {
            IDENTITY_,
            GAMMA,
            COLORPOLYFIT,
            COLORLOGPOLYFIT,
            GRAYPOLYFIT,
            GRAYLOGPOLYFIT
        };

        /* Polyfit model */
        class Polyfit {
        public:
            int deg;
            cv::Mat p;
            Polyfit() {};
            Polyfit(cv::Mat s, cv::Mat d, int deg) :deg(deg) {
                int npoints = s.checkVector(1);           
                int nypoints = d.checkVector(1);
                cv::Mat_<double> srcX(s), srcY(d);
                cv::Mat_<double> A = cv::Mat_<double>::ones(npoints, deg + 1);
                for (int y = 0; y < npoints; ++y)
                {
                    for (int x = 1; x < A.cols; ++x)
                    {
                        A.at<double>(y, x) = srcX.at<double>(y) * A.at<double>(y, x -1);
                    }
                }
                cv::solve(A, srcY, p, DECOMP_SVD);
            }
            virtual ~Polyfit() {};
           
            cv::Mat operator()(cv::Mat inp) {
                return _elementwise(inp, [this](double a)->double {return _from_ew(a); });
            };
        private:
            double _from_ew(double x) {
                double res = 0;
                for (int d = 0; d <= deg; d++) {
                    res += pow(x, d) * p.at<double>(d, 0);
                }
                return res;
            };
        };

        /* Logpolyfit model */
        class LogPolyfit {
        public:
            int deg;
            Polyfit p;
            LogPolyfit() {};
            LogPolyfit(cv::Mat s, cv::Mat d, int deg) :deg(deg) {
                cv::Mat mask_ = (s > 0) & (d > 0);
               // mask_.convertTo(mask_, CV_64F);
                cv::Mat src_, dst_, s_, d_;
                src_ = mask_copyto(s, mask_);
                dst_ = mask_copyto(d, mask_);
                log(src_, s_);
                log(dst_, d_);
                p = Polyfit(s_, d_, deg);
            }
            virtual ~LogPolyfit() {};

            cv::Mat operator()(cv::Mat inp) {
                cv::Mat mask_ = inp >= 0;
                cv::Mat y, y_, res;
                log(inp, y);
                y = p(y);
                exp(y, y_);
                y_.copyTo(res, mask_);
                return res;
            };
        };

        /* linearization base */
        class Linear
        {
        public:
            Linear() {};
            virtual ~Linear() {};

            // inference
            virtual cv::Mat linearize(cv::Mat inp) { return inp; };
            // evaluate linearization model
            virtual void value(void) {};
        };

        /* make no change */
        class Linear_identity : public Linear
        {
        public:
            using Linear::Linear;
        };

        /* gamma correction */
        class Linear_gamma : public Linear
        {
        public:
            double gamma;
            Linear_gamma(double gamma) :gamma(gamma) {};
            cv::Mat linearize(cv::Mat inp) {
                return gamma_correction(inp, gamma);
            };
        };

        /* grayscale polynomial fitting */
        template <class T>
        class Linear_gray :public Linear {
        public:
            int deg;
            T p;
            Linear_gray(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs) :deg(deg) {
                dst.get_gray();
                Mat lear_gray_mask = mask & dst.grays;
                // the grayscale function is approximate for src is in relative color space;
                src = rgb2gray(mask_copyto(src, lear_gray_mask));
                cv::Mat dst_ = mask_copyto(dst.toGray(cs.io), lear_gray_mask);
                calc(src, dst_);
            }
            // monotonically increase is not guaranteed
            void calc(cv::Mat src, cv::Mat dst) {
                p = T(src, dst, deg);
            };
            cv::Mat linearize(cv::Mat inp) {
                return p(inp);
            };
        };

        /* fitting channels respectively */
        template <class T>
        class Linear_color :public Linear {
        public:
            int deg;
            T pr;
            T pg;
            T pb;
            Linear_color(int deg, cv::Mat src_, Color dst, cv::Mat mask, RGB_Base_ cs) :deg(deg) {
              //  dst.get_gray();
               // std::cout << "mask" << mask << std::endl;
              //  mask = mask & dst.grays;
                Mat src = mask_copyto(src_, mask);
                cv::Mat dst_ = mask_copyto(dst.to(*cs.l).colors, mask);
                calc(src, dst_);
            }

            // monotonically increase is not guaranteed
            void calc(cv::Mat src, cv::Mat dst) {
                
                cv::Mat sChannels[3];
                cv::Mat dChannels[3];
                split(src, sChannels);
                split(dst, dChannels);
                std::cout << "sChannels[0]" << sChannels[0] << std::endl;
                pr = T(sChannels[0], dChannels[0], deg);
                pg = T(sChannels[1], dChannels[1], deg);
                pb = T(sChannels[2], dChannels[2], deg);
            };

            cv::Mat linearize(cv::Mat inp) {
                cv::Mat channels[3];
                split(inp, channels);
                std::vector<cv::Mat> channel;
                cv::Mat res;
                channel.push_back(pr(channels[0]));
                channel.push_back(pg(channels[1]));
                channel.push_back(pb(channels[2]));
                merge(channel, res);
                return res;
            };
        };

        /* get linearization method */
        Linear* get_linear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs, LINEAR_TYPE lineat_type)
        {
            Linear* p = new Linear();
            switch (lineat_type)
            {
            case cv::ccm::IDENTITY_:
                p = new Linear_identity();
                return p;
                break;
            case cv::ccm::GAMMA:
                p = new Linear_gamma(gamma);
                return p;
                break;
            case cv::ccm::COLORPOLYFIT:
                p = new Linear_color<Polyfit>(deg, src, dst, mask, cs);
                return p;
                break;
            case cv::ccm::COLORLOGPOLYFIT:
                p = new Linear_color<LogPolyfit>(deg, src, dst, mask, cs);
                return p;
                break;
            case cv::ccm::GRAYPOLYFIT:
                p = new Linear_gray<Polyfit>(deg, src, dst, mask, cs);
                return p;
                break;
            case cv::ccm::GRAYLOGPOLYFIT:
                p = new Linear_gray<LogPolyfit>(deg, src, dst, mask, cs);
                return p;
                break;
            }
        };
    }
}


#endif