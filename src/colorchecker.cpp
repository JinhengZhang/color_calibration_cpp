#include "colorchecker.h"

ColorChecker::ColorChecker(cv::Mat color, string colorspace, IO io_, cv::Mat whites) 
{
	if (colorspace == "LAB")
	{
		this->lab = color;
		this->io = io_;
	}
	else
	{
		this->rgb = color;
		this->cs = getColorspace(colorspace);
	}
	vector<double> white_m(color.rows, 1);
	if (!whites.empty())
	{
		for (int i = 0; i < whites.cols; i++)
		{
			white_m[whites.at<double>(0, i)] = 0;
		}
		this->white_mask = cv::Mat(white_m, true);
	}
	color_mask = cv::Mat(white_m, true);
}

ColorCheckerMetric::ColorCheckerMetric(ColorChecker colorchecker, string colorspace, IO io_)
{
	this->cc = colorchecker;
	this->cs = getColorspace(colorspace);
	this->io = io_;
	if (!this->cc.lab.empty())
	{
		this->lab = lab2lab(this->cc.lab, cc.io, io_);
		this->xyz = lab2xyz(lab, io_);
		this->rgbl = this->cs->xyz2rgbl(this->xyz, io_);
		this->rgb = cs->rgbl2rgb(this->rgbl);
	}
	else
	{
		this->rgb = cs->xyz2rgb(cc.cs->rgb2xyz(cc.rgb, D65_2), D65_2);
		this->rgbl = cs->rgb2rgbl(rgb);
		this->xyz = cs->rgbl2xyz(rgbl, io);
		this->lab = xyz2lab(xyz, io);
	}
	this->grayl = xyz2grayl(xyz);
	this->white_mask = cc.white_mask;
	this->color_mask = cc.color_mask;
}
