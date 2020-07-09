#include "colorchecker.h"


ColorChecker::ColorChecker(Mat color, string colorspace, IO io_, Mat whites) {
	if (colorspace == "lab")
	{
		lab = color;
		io = io_;
	}
	else
	{
		rgb = color;
		cs = get_colorspace(colorspace);
	}

	vector<bool> white_m(color.rows, false);
	if (!whites.empty())
	{
		for (int i = 0; i < whites.cols; i++) 
		{
			white_m[whites.at<double>(0, i)] = true;
		}
	}
	Mat white_mask = Mat(white_m);

	color_mask = ~white_mask;
}

ColorCheckerMetric::ColorCheckerMetric(ColorChecker colorchecker, string colorspace, IO io_)
{
	cc = colorchecker;
	cs = get_colorspace(colorspace);
	io = io_;
	if (!cc.lab.empty())
	{
		lab = lab2lab(cc.lab, cc.io, io_);
		xyz = lab2xyz(lab, io_);
		rgbl = cs->xyz2rgbl(xyz, io_);
		rgb = cs->rgbl2rgb(rgbl);
	}
	else
	{
		rgb = cs->xyz2rgb(cc.cs->rgb2xyz(cc.rgb, IO("D65", 2)), IO("D65", 2));
		rgbl = cs->rgb2rgbl(rgb);
		xyz = cs->rgbl2xyz(rgbl, io);
		lab = xyz2lab(xyz, io);
	}
	grayl = xyz2grayl(xyz);
	white_mask = cc.white_mask;
	color_mask = cc.color_mask;
}

