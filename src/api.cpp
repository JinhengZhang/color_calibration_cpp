#include "api.h"

CCM_3x3* get_ccm(string ccm_shape, Mat src_, Mat dst, string dst_colorspace, string dst_illuminant, int dst_observer, Mat dst_whites, string colorchecker, vector<double> saturated_threshold, string colorspace, string linear_, float gamma, float deg, string distance_, string dist_illuminant, int dist_observer, Mat weights_list, double weights_coeff, bool weights_color, string initial_method, double xtol_, double ftol_)
{
    CCM_3x3* p = new CCM_3x3;
    if (ccm_shape == "CCM_3x3") {
        p = new CCM_3x3;
    }
    else if (ccm_shape == "CCM_4x3") {
        p = new CCM_4x3;
    }
    return p;
}
