#include "linearize.h"

namespace cv {
    namespace ccm {
        Polyfit::Polyfit(cv::Mat s, cv::Mat d, int deg) :deg(deg) {
            int npoints = s.checkVector(1);
            int nypoints = d.checkVector(1);
            cv::Mat_<double> srcX(s), srcY(d);
            cv::Mat_<double> A = cv::Mat_<double>::ones(npoints, deg + 1);
            for (int y = 0; y < npoints; ++y)
            {
                for (int x = 1; x < A.cols; ++x)
                {
                    A.at<double>(y, x) = srcX.at<double>(y) * A.at<double>(y, x - 1);
                }
            }
            cv::solve(A, srcY, p, DECOMP_SVD);
        };

        double Polyfit::_from_ew(double x) {
            double res = 0;
            for (int d = 0; d <= deg; d++) {
                res += pow(x, d) * p.at<double>(d, 0);
                return res;
            }
        };

        cv::Mat Polyfit::operator()(cv::Mat inp) {
            return _elementwise(inp, [this](double a)->double {return _from_ew(a); });
        }
        
        /*
        cv::Mat Polyfit::operator()(cv::Mat inp) {
            cv::Mat res_polyfit(inp.size(), inp.type());//
            for (int i = 0; i < inp.rows; i++) {
                for (int j = 0; j < inp.cols; j++) {
                    double res = 0;
                    for (int d = 0; d <= deg; d++) {
                        res += pow(inp.at<double>(i, j), d) * p.at<double>(d, 0);
                        res_polyfit.at<double>(i, j) = res;
                    }
                }
            }
            return res_polyfit;
        }
        */
        
        LogPolyfit::LogPolyfit(cv::Mat s, cv::Mat d, int deg) :deg(deg) {
            cv::Mat mask_ = (s > 0) & (d > 0);
            mask_.convertTo(mask_, CV_64F);
            cv::Mat src_, dst_;
            s = mask_copyto(s, mask_);//
            d = mask_copyto(d, mask_);
            cv::Mat s, d;
            log(src_, s);
            log(dst_, d);
            p = Polyfit(s, d, deg);
        }

        cv::Mat LogPolyfit::operator()(cv::Mat inp) {
            cv::Mat mask_ = inp >= 0;
            cv::Mat y;
            log(inp, y);
            y = p(y);
            cv::Mat y_;
            exp(y, y_);
            cv::Mat res;
            y_.copyTo(res, mask_);
            return res;
        }

        cv::Mat Linear_gamma::linearize(cv::Mat inp) {
            return gamma_correction(inp, gamma);
        }

        template <class T>
        Linear_gray<T>::Linear_gray(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs) :deg(deg) {
            dst.get_gray();
            mask = mask & dst.grays;
            src = rgb2gray(mask_copyto(src, mask));
            cv::Mat dst_ = mask_copyto(dst.toGray(cs.io), mask);
            calc(src, dst_);
        }

        template <class T>
        void Linear_gray<T>::calc(cv::Mat src, cv::Mat dst) {
            p = T(src, dst, deg);
        }

        template <class T>
        cv::Mat Linear_gray<T>::linearize(cv::Mat inp) {
            return p(inp);
        }

        template <class T>
        Linear_color<T>::Linear_color(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs) :deg(deg) {
            dst.get_gray();
            mask = mask & dst.grays;
            src = mask_copyto(src, mask);
            cv::Mat dst_ = mask_copyto(dst.to(*cs.l).colors, mask);
            calc(src, dst_);
        }

        template <class T>
        void Linear_color<T>::calc(cv::Mat src, cv::Mat dst) {
            cv::Mat sChannels[3];
            cv::Mat dChannels[3];
            split(src, sChannels);
            split(dst, dChannels);
            pr = T(sChannels[0], dChannels[0], deg);
            pg = T(sChannels[1], dChannels[1], deg);
            pb = T(sChannels[2], dChannels[2], deg);
        }

        template <class T>
        cv::Mat Linear_color<T>::linearize(cv::Mat inp) {
            cv::Mat channels[3];
            split(inp, channels);
            std::vector<cv::Mat> channel;
            cv::Mat res;
            channel.push_back(pr(channels[0]));
            channel.push_back(pg(channels[1]));
            channel.push_back(pb(channels[2]));
            merge(channel, res);
            return res;
        }

        /*
        Linear get_linear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs, LINEAR_TYPE linear_type) {
            //Linear p();
            switch (linear_type) {
            case cv::ccm::IDENTITY:
                Linear_identity p();
                return p;
                break;
            case cv::ccm::GAMMA:
                Linear_gamma p(gamma);
                return p;
                break;
            case cv::ccm::COLORPOLYFIT:
                Linear_color<Polyfit> p(deg, src, dst, mask, cs);
                return p;
                break;
            case cv::ccm::COLORLOGPOLYFIT:
                Linear_color<LogPolyfit> p(deg, src, dst, mask, cs);
                return p;
                break;
            case cv::ccm::GRAYPOLYFIT:
                Linear_gray<Polyfit> p(deg, src, dst, mask, cs);
                return p;
                break;
            case cv::ccm::GRAYLOGPOLYFIT:
                Linear_gray<LogPolyfit> p(deg, src, dst, mask, cs);
                return p;
                break;
            }
        }*/
        
        Linear* get_linear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs, LINEAR_TYPE linear_type) {
            Linear* p = new Linear();
            switch (linear_type) {
            case cv::ccm::IDENTITY:
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
        }
    }
}
