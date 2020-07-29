#include "colorchecker.h"

/*
	the colorchecker;
	color: reference colors of colorchecker;
	colorspace: 'LAB' or some 'RGB' color space;
	io: only valid if colorspace is 'LAB';
	whites: the indice list of gray colors of the reference colors;
*/
ColorChecker::ColorChecker(cv::Mat color, string colorspace, IO io_, cv::Mat whites) 
{
	// color and correlated color space
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
	// white_mask& color_mask
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

/* the colorchecker adds the color space for conversion for color distance; */
ColorCheckerMetric::ColorCheckerMetric(ColorChecker colorchecker, string colorspace, IO io_)
{
	// colorchecker
	this->cc = colorchecker;

	// color space
	this->cs = getColorspace(colorspace);
	this->io = io_;

	// colors after conversion
	if (this->cc.lab.data)
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

	// white_mask & color_mask
	this->white_mask = cc.white_mask;
	this->color_mask = cc.color_mask;
}
