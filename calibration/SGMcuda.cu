#include <stdlib.h>
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ximgproc/disparity_filter.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <libsgm.h>

using namespace cv::ximgproc;
using namespace std;
//错误结果输出
#define ASSERT_MSG(expr, msg) \
	if (!(expr)) { \
		std::cerr << msg << std::endl; \
		std::exit(EXIT_FAILURE); \
	} \

//改写sample使其适用于每一帧双目图像(固定匹配算法内部参数)
cv::Mat SGMcuda(cv::Mat left, cv::Mat right)
{
	const int disp_size = 128;
	const int P1 = 10;
	const int P2 = 120;
	const float uniqueness = 95;
	const int num_paths = 8;
	const int min_disp = 0;
	const int LR_max_diff = 2;


	//对图像读取正确性的检查
	ASSERT_MSG(!left.empty() && !right.empty(), "imread failed.");
	ASSERT_MSG(left.size() == right.size() && left.type() == right.type(), "input images must be same size and type.");
	ASSERT_MSG(left.type() == CV_8U || left.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
	ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");
	ASSERT_MSG(num_paths == 4 || num_paths == 8, "number of scanlines must be 4 or 8.");

	const sgm::PathType path_type = num_paths == 8 ? sgm::PathType::SCAN_8PATH : sgm::PathType::SCAN_4PATH;
	const int input_depth = left.type() == CV_8U ? 8 : 16;
	const int output_depth = 16;

	//创建sgm对象的参数
	const sgm::StereoSGM::Parameters param(P1, P2, uniqueness, false, path_type, min_disp, LR_max_diff);

	sgm::StereoSGM ssgm(left.cols, left.rows, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);

	cv::Mat disparity(left.size(), CV_16S);

	//计算视差
	ssgm.execute(left.data, right.data, disparity.data);

	// create mask for invalid disp
	cv::Mat mask = disparity == ssgm.get_invalid_disparity();

	// show image
	cv::Mat disparity_8u, disparity_color;
	disparity.convertTo(disparity_8u, CV_8U, 255.0 / disp_size);

	//对视差图进行后处理(滤波)
	cv::Mat rightDisparity(right.size(), CV_16S);
	cv::Mat flippedRightDisparity(right.size(), CV_16S);

	cv::Mat flippedLeft(left.size(), CV_8U);
	cv::Mat flippedRight(right.size(), CV_8U);

	cv::flip(left, flippedLeft, 1);
	cv::flip(right, flippedRight, 1);
	sgm::StereoSGM ssgm_right(left.cols, left.rows, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_HOST2HOST, param);
	ssgm_right.execute(flippedRight.data, flippedLeft.data, flippedRightDisparity.data);
	flip(flippedRightDisparity, rightDisparity, 1);
	rightDisparity *= -16;
	cv::Mat rightDisparity_8u;
	rightDisparity.convertTo(rightDisparity_8u, CV_8U, 255.0 / disp_size);

	cv::Ptr<DisparityWLSFilter> wlsFilter = createDisparityWLSFilterGeneric(false);
	cv::Mat filteredDisp, filteredDisp8u;
	wlsFilter->filter(disparity, left, filteredDisp, rightDisparity);
	filteredDisp.convertTo(filteredDisp8u, CV_8U, 255.0 / disp_size);

	//disparity /= sgm::StereoSGM::SUBPIXEL_SCALE;
	//disparity.convertTo(disparity_8u, CV_8U, 255.0 / disp_size);
	cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	disparity_color.setTo(cv::Scalar(0, 0, 0), mask);
	disparity_8u.setTo(0, mask);
	//imshow("SGMdisparity", disparity_8u);
	//imshow("SGMcolordisparity",disparity_color);
	imshow("SGMfilterdisparity", filteredDisp8u);
	//imshow("SGMfilterdisparity16s", filteredDisp);

	//cout << filteredDisp.type() << endl;
	return filteredDisp;
}