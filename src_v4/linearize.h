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


        class Polyfit {
        public:
            int deg;
            cv::Mat p;
            Polyfit() {};
            Polyfit(cv::Mat s, cv::Mat d, int deg);
            double _from_ew(double x);
            cv::Mat operator()(cv::Mat inp);
        };


        class LogPolyfit {
        public:
            int deg;
            Polyfit p;
            LogPolyfit() {};
            LogPolyfit(cv::Mat s, cv::Mat d, int deg);
            cv::Mat operator()(cv::Mat inp);
        };


        class Linear
        {
        public:
            Linear() {};
            virtual cv::Mat linearize(cv::Mat inp) { return inp; };
            virtual void value(void) {};
        };


        class Linear_identity : public Linear
        {
        public:
            using Linear::Linear;
        };


        class Linear_gamma : public Linear
        {
        public:
            double gamma;
            Linear_gamma(double gamma) :gamma(gamma) {};
            cv::Mat linearize(cv::Mat inp);
        };


        template <class T>
        class Linear_gray :public Linear {
        public:
            int deg;
            T p;
            Linear_gray(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs);
            void calc(cv::Mat src, cv::Mat dst);
            cv::Mat linearize(cv::Mat inp);
        };


        template <class T>
        class Linear_color :public Linear {
        public:
            int deg;
            T pr;
            T pg;
            T pb;
            Linear_color(int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs);
            void calc(cv::Mat src, cv::Mat dst);
            cv::Mat linearize(cv::Mat inp);
        };

        Linear* get_linear(double gamma, int deg, cv::Mat src, Color dst, cv::Mat mask, RGB_Base_ cs, LINEAR_TYPE lineat_type);

    }
}

#endif
