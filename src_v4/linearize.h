#ifndef LINEARIZE_H
#define LINEARIZE_H

#include "utils.h"
#include "color.h"

namespace cv {
    namespace ccm {
        enum LINEAR_TYPE {
            IDENTITY,
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
            Polyfit(cv::Mat s, cv::Mat d, int deg);
            double _from_ew(double x);
            cv::Mat operator()(cv::Mat inp);
        };

        /* Logpolyfit model */
        class LogPolyfit {
        public:
            int deg;
            Polyfit p;
            LogPolyfit() {};
            LogPolyfit(cv::Mat s, cv::Mat d, int deg);
            cv::Mat operator()(cv::Mat inp);
        };

        /* linearization base */
        class Linear
        {
        public:
            Linear() {};
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
            cv::Mat linearize(cv::Mat inp);
        };

        /* grayscale polynomial fitting */
        template <class T>
        class Linear_gray :public Linear {
        public:
            int deg;
            T p;
            Linear_gray(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs);
            // monotonically increase is not guaranteed
            void calc(cv::Mat src, cv::Mat dst);
            cv::Mat linearize(cv::Mat inp);
        };

        /* fitting channels respectively */
        template <class T>
        class Linear_color :public Linear {
        public:
            int deg;
            T pr;
            T pg;
            T pb;
            Linear_color(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs);
            // monotonically increase is not guaranteed
            void calc(cv::Mat src, cv::Mat dst);
            cv::Mat linearize(cv::Mat inp);
        };

        /* get linearization method */
        Linear* get_linear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs, LINEAR_TYPE lineat_type);

    }
}

#endif
