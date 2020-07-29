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
		lab = color;
		io = io_;
	}
	else
	{
		rgb = color;
		cs = getColorspace(colorspace);
	}
	// white_mask& color_mask
	vector<double> white_m(color.rows, 1);
	if (whites.data)
	{
		for (int i = 0; i < whites.cols; i++)
		{
			white_m[whites.at<double>(0, i)] = 0;
		}
		white_mask = cv::Mat(white_m, true);
	}
	color_mask = cv::Mat(white_m, true);
}

/* the colorchecker adds the color space for conversion for color distance; */
ColorCheckerMetric::ColorCheckerMetric(ColorChecker colorchecker, string colorspace, IO io_)
{
	// colorchecker
	cc = colorchecker;

	// color space
	cs = getColorspace(colorspace);
	io = io_;

	// colors after conversion
	if (cc.lab.data)
	{
		lab = lab2lab(cc.lab, cc.io, io_);
		xyz = lab2xyz(lab, io_);
		rgbl = cs->xyz2rgbl(xyz, io_);
		rgb = cs->rgbl2rgb(rgbl);
	}
	else
	{
		rgb = cs->xyz2rgb(cc.cs->rgb2xyz(cc.rgb, D65_2), D65_2);
		rgbl = cs->rgb2rgbl(rgb);
		xyz = cs->rgbl2xyz(rgbl, io);
		lab = xyz2lab(xyz, io);
	}
	grayl = xyz2grayl(xyz);

	// white_mask & color_mask
	white_mask = cc.white_mask;
	color_mask = cc.color_mask;
}
