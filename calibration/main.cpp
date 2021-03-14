
#include <vector>  
#include <string>  
#include <numeric>
#include <algorithm>  
#include <iostream>  
#include <iterator>  
#include <stdio.h>  
#include <stdlib.h>  
#include <ctype.h>  
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/stereo.hpp>
#include <opencv2/cudastereo.hpp>

#include "LaneDetecter.h"
#include "lsd.h"
#include "header.h"
#include "image_enhancement.h"
#include "fastbilateralfilter.h"
#include "fastbilateral2.h"


extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

using namespace std;
using namespace cv;                                         //依旧很长的开头


bool hasCalibData = true;


const int imageWidth = 1279;                             //摄像头的分辨率  
const int imageHeight = 960;
const int boardWidth = 10;                               //横向的角点数目  
const int boardHeight = 7;                              //纵向的角点数据  
const int boardCorner = boardWidth * boardHeight;       //总的角点数据  
const int frameNumber = 20;                             //相机标定时需要采用的图像帧数  
const int squareSize = 59;                              //标定板黑白格子的大小 单位mm  
const Size boardSize = Size(boardWidth, boardHeight);   //标定板的总内角点  
Size imageSize = Size(imageWidth, imageHeight);

Mat R, T, E, F;                                                  //R 旋转矢量 T平移矢量 E本征矩阵 F基础矩阵  
vector<Mat> rvecs;                                        //旋转向量  
vector<Mat> tvecs;                                        //平移向量  
vector<vector<Point2f>> imagePointL;                    //左边摄像机所有照片角点的坐标集合  
vector<vector<Point2f>> imagePointR;                    //右边摄像机所有照片角点的坐标集合  
vector<vector<Point3f>> objRealPoint;                   //各副图像的角点的实际物理坐标集合  

vector<Point2f> cornerL;                              //左边摄像机某一照片角点坐标集合  

vector<Point2f> cornerR;                              //右边摄像机某一照片角点坐标集合  

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;

Mat Rl, Rr, Pl, Pr, Q;                                  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q (下面有具体的含义解释）   
Mat mapLx, mapLy, mapRx, mapRy;                         //映射表  
Rect validROIL, validROIR;                              //图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
//Mat cameraMatrixL = Mat::eye(3, 3, CV_64F);
//Mat cameraMatrixR = Mat::eye(3, 3, CV_64F);
//Mat distCoeffL = Mat::zeros(5, 1, CV_64F);
//Mat distCoeffR = Mat::zeros(5, 1, CV_64F);

Mat rectifyImageL2, rectifyImageR2;

string pro_dirMain = "D:/darknet-master/build/darknet/x64/"; //项目根目录
String modelConfigurationDefaultMain = pro_dirMain + "cfg/yolov3.cfg";
String modelWeightsDefaultMain = pro_dirMain + "yolov3.weights";
string image_pathMain = "D:/text/02.jpg";
string classesFileDefaultMain = pro_dirMain + "data/coco.names";// "coco.names";
string video_pathMain = "D:/Gymshark.avi";


//立体匹配所需
int mindisparity = 0;
int ndisparities = 64;
int SADWindowSize = 5;
Ptr<StereoSGBM> sgbm = StereoSGBM::create();

Mat xyz;
//Ptr<StereoBM> bm = StereoBM::create(16, 9);
Point origin;
Rect selection;
bool selectObject = false;
//int blockSize = 0, uniquenessRatio = 0, numDisparities = 0;
Ptr<StereoBM> bm = StereoBM::create(16, 9);
int blockSize = 3, uniquenessRatio = 16, numDisparities = 3;

Ptr<cuda::StereoBM> bmcuda;

/*
事先标定好的左相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/

Mat cameraMatrixL = (Mat_<double>(3, 3) << 856.2098864229069, 0, 612.8017954183415,
	0, 860.3241625151114, 444.0452190891914,
	0, 0, 1);                                                                           //这时候就需要你把左右相机单目标定的参数给写上
//获得的畸变参数
Mat distCoeffL = (Mat_<double>(5, 1) << -0.2751704240600508, 3.61771723826387, 0.006850903418294842, 0.0006043288374723176, -21.14342876340745);

/*
事先标定好的右相机的内参矩阵
fx 0 cx
0 fy cy
0 0  1
*/

Mat cameraMatrixR = (Mat_<double>(3, 3) << 862.9542060459133, 0, 602.2804058247383,
	0, 866.6683711126871, 434.1267646098479,
	0, 0, 1);
Mat distCoeffR = (Mat_<double>(5, 1) << -0.1404158961254861, 1.366924877217865, 0.00665465738444818, -0.002205552069359751, -7.133244919058638);


vector<Mat>roi_rects;
vector<Mat>road_img;
Mat deal_img; //保存旋转校正后的道路图像
vector<Mat> rotmats;
vector<Rect> speedboxes;

float manualselectdepth = 0.0;
float final_distance = 0.0; 

Mat rotdip8u;
Mat rotsrc;

int baseLine;
Size labelSize;
int top;
Rect speedbox;
//手动选取测距点
/*计算标定板上模块的实际物理坐标*/
void calRealPoint(vector<vector<Point3f>>& obj, int boardwidth, int boardheight, int imgNumber, int squaresize)
{
	vector<Point3f> imgpoint;
	for (int rowIndex = 0; rowIndex < boardheight; rowIndex++)
	{
		for (int colIndex = 0; colIndex < boardwidth; colIndex++)
		{
			imgpoint.push_back(Point3f(rowIndex * squaresize, colIndex * squaresize, 0));
		}
	}
	for (int imgIndex = 0; imgIndex < imgNumber; imgIndex++)
	{
		obj.push_back(imgpoint);
	}
}


void outputCameraParam(void)
{
	//保存数据
	//输出数据
	FileStorage fs("intrinsics.yml", FileStorage::WRITE);  //文件存储器的初始化
	if (fs.isOpened())
	{
		fs << "cameraMatrixL" << cameraMatrixL << "cameraDistcoeffL" << distCoeffL << "cameraMatrixR" << cameraMatrixR << "cameraDistcoeffR" << distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
		cout << "R = " << R << "T = " << T << "E = " << E << "F = " << F << endl;
	}
	else
	{
		cout << "Error: can not save the intrinsics!!!!!" << endl;
	}


	fs.open("extrinsics.yml", FileStorage::WRITE);
	if (fs.isOpened())
	{
		fs << "R" << R << "T" << T << "Rl" << Rl << "Rr" << Rr << "Pl" << Pl << "Pr" << Pr << "Q" << Q << "mapLx" << mapLx << "mapLy" << mapLy << "mapRx" << mapRx << "mapRy" << mapRy <<"validROIL" << validROIL << "validROIR" << validROIR;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
		fs.release();
	}
	else
		cout << "Error: can not save the extrinsic parameters\n";
}

void inputCameraParm(void)
{
	FileStorage fs("intrinsics.yml", FileStorage::READ);  //文件存储器的初始化
	if (fs.isOpened())
	{
		fs["cameraMatrixL"] >> cameraMatrixL;
		fs["cameraDistcoeffL"] >> distCoeffL;
		fs["cameraMatrixR"] >> cameraMatrixR;
		fs["cameraDistcoeffR"] >> distCoeffR;
		fs.release();
		cout << "cameraMatrixL=:" << cameraMatrixL << endl << "cameraDistcoeffL=:" << distCoeffL << endl << "cameraMatrixR=:" << cameraMatrixR << endl << "cameraDistcoeffR=:" << distCoeffR << endl;
	}
	else
	{
		cout << "Error: can not read the intrinsics!!!!!" << endl;
	}


	fs.open("extrinsics.yml", FileStorage::READ);
	if (fs.isOpened())
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["Rl"] >> Rl;
		fs["Rr"] >> Rr;
		fs["Pl"] >> Pl;
		fs["Pr"] >> Pr;
		fs["Q"] >> Q;
		fs["mapLx"] >> mapLx;
		fs["mapLy"] >> mapLy;
		fs["mapRx"] >> mapRx;
		fs["mapRy"] >> mapRy;
		fs["validROIL"] >> validROIL;
		fs["validROIR"] >> validROIR;
		cout << "R=" << R << endl << "T=" << T << endl << "Rl=" << Rl << endl << "Rr=" << Rr << endl << "Pl=" << Pl << endl << "Pr=" << Pr << endl << "Q=" << Q << endl;
		fs.release();
	}
	else
		cout << "Error: can not read the extrinsic parameters\n";
}
void  stereo_SGBM_match(int, void*)
{

	int P1 = 8 * rectifyImageL2.channels() * SADWindowSize* SADWindowSize;
	int P2 = 32 * rectifyImageL2.channels() * SADWindowSize* SADWindowSize;
	sgbm->setP1(P1);
	sgbm->setP2(P2);
	sgbm->setPreFilterCap(15);
	sgbm->setUniquenessRatio(6);
	sgbm->setSpeckleRange(2);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setNumDisparities(128);
	//sgbm->setMode(cv::StereoSGBM::MODE_HH);
	Mat disp, disp8U;
	sgbm->compute(rectifyImageL2, rectifyImageR2, disp);
	disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值
	disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示
	//normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);
	disp.convertTo(disp8U, CV_8U, 255 / (ndisparities*16.));
	reprojectImageTo3D(disp, xyz, Q);
	xyz = xyz * 16;
	imshow("disparity", xyz);

}
int SGBM_SADWindowSize = 15;
int SGBM_numberOfDisparities = 128;
int SGBM_uniquenessRatio = 15;
Mat Match_SGBM(Mat left, Mat right)//SGBM匹配算法，输入left、right为灰度图，输出disp8为灰度图
{
	
	
	int SADWindowSize = SGBM_SADWindowSize;//主要影响参数
	int numberOfDisparities = SGBM_numberOfDisparities;//主要影响参数
	int uniquenessRatio = SGBM_uniquenessRatio;//主要影响参数，越大误匹配越小，不能匹配区域越多

	int cn = left.channels();
	sgbm->setPreFilterCap(63);
	sgbm->setBlockSize(SGBM_SADWindowSize);
	sgbm->setP1(8 * cn*SADWindowSize*SADWindowSize);
	sgbm->setP2(32 * cn*SADWindowSize*SADWindowSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(uniquenessRatio);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);
	Mat disp, disp8;
	int64 t = getTickCount();
	sgbm->compute(left, right, disp);
	t = getTickCount() - t;
	cout << "SGBM time: " << t * 1000 / getTickFrequency() << "ms，also " << t / getTickFrequency() << " s" << endl;
	disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	return disp8;
}


void stereo_match(int, void*)
{
	/*
	Mat testimgL = imread("./im2.png");
	Mat testimgR = imread("./im6.png");
	Mat disptest, disp8test;
	cvtColor(testimgL, testimgL, COLOR_BGR2GRAY);
	cvtColor(testimgR, testimgR, COLOR_BGR2GRAY);
	*/
	bm->setPreFilterType(1);
	bm->setBlockSize(2 * blockSize + 5);     //SAD窗口大小，5~21之间为宜
	bm->setROI1(validROIL);  //设置了ROI后，只会计算出两图公共区域的视差图像，因而可以在立体校正后，若图像边缘畸变不大，则可不设置ROI区域
	bm->setROI2(validROIR);
	bm->setPreFilterCap(31);
	bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
	bm->setNumDisparities(numDisparities * 16 + 16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(-1);
	Mat disp, disp8;
	Mat gray_rectifyImageL2(rectifyImageL2);
	Mat gray_rectifyImageR2(rectifyImageR2);

	//cvtColor(rectifyImageL2, gray_rectifyImageL2, COLOR_BGR2GRAY);
	//cvtColor(rectifyImageR2, gray_rectifyImageR2, COLOR_BGR2GRAY);

	bm->compute(gray_rectifyImageL2, gray_rectifyImageR2, disp);//输入图像必须为灰度图

	//输出视差矩阵
	/*
	for (int row = 0; row != disp.rows;++row)
	{
		for (int col = 0; col != disp.cols; ++col)
			cout << float(disp.at<short>(row, col))/16 << " ";
		cout << endl;
	}
	*/
	disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
	reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("BMdisparity", disp8);

	/*
	bm->compute(testimgL, testimgR, disptest);
	disptest.convertTo(disp8test, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));
	imshow("BMtestdisparity", disp8test);
	*/
}

void stereo_match1(int, void*)
{
	sgbm->setPreFilterCap(63);
	int sgbmWinSize = 5;//根据实际情况自己设定
	int NumDisparities = 64;//根据实际情况自己设定
	int UniquenessRatio = 6;//根据实际情况自己设定
	sgbm->setBlockSize(sgbmWinSize);
	int cn = rectifyImageL2.channels();

	//取roi区域尝试计算视差图，以缩短计算时间
	//Rect rectifyImagerect = Rect(120, 360, 900, 600);
	//Mat rectifyImageL2Roi = rectifyImageL2(rectifyImagerect);
	//Mat rectifyImageR2Roi = rectifyImageR2(rectifyImagerect);
	//imshow("left_roi", rectifyImageL2Roi);
	//imshow("rightroi", rectifyImageR2Roi);



	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(NumDisparities);
	sgbm->setUniquenessRatio(UniquenessRatio);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(2);
	sgbm->setDisp12MaxDiff(1);
	sgbm->setMode(StereoSGBM::MODE_SGBM);
	Mat disp, dispf, disp8;
	sgbm->compute(rectifyImageL2, rectifyImageR2, disp);
	//去黑边
	Mat img1p, img2p;
	copyMakeBorder(rectifyImageL2, img1p, 0, 0, NumDisparities, 0, BORDER_REPLICATE);
	copyMakeBorder(rectifyImageR2, img2p, 0, 0, NumDisparities, 0, BORDER_REPLICATE);
	dispf = disp.colRange(NumDisparities, img2p.cols - NumDisparities);

	dispf.convertTo(disp8, CV_8U, 255 / (NumDisparities *16.));
	reprojectImageTo3D(dispf, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
	xyz = xyz * 16;
	imshow("disparity", disp8);
	//Mat color(dispf.size(), CV_8UC3);
	//GenerateFalseMap(disp8, color);//转成彩图
	//imshow("disparity", color);
	//saveXYZ("xyz.xls", xyz);
}

extern Mat SGMcuda(cv::Mat left, cv::Mat right);
extern vector<string> loadModel(string modelWeights, string modelConfiguration, string classesFile);
extern void objectDetection(Mat leftImg, vector<int>& indices, vector<int>& classIds, vector<Rect>& boxes);


//根据视差图计算前方减速带距离本车距离时，计算量较大，计算缓慢

Point3f uv2xyz(Point2f uvLeft, Point2f uvRight)
{
	//     [u1]      [xw]                      [u2]      [xw]
	//zc1 *|v1| = Pl*[yw]                  zc2*|v2| = P2*[yw]
	//     [ 1]      [zw]                      [ 1]      [zw]
	//               [1 ]    
	Mat mLeftRotation = (Mat_<double>(3, 3) << 1, 0, 0,
		0, 1, 0,
		0, 0, 1);
	Mat mLeftTranslation = (Mat_<double>(3, 1) << 0, 0, 0);

	Mat mLeftRT = Mat(3, 4, CV_64FC1);//左相机RT矩阵
	hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
	Mat mLeftIntrinsic = cameraMatrixL;
	Mat mLeftP = mLeftIntrinsic * mLeftRT;
	cout<<"左相机P矩阵 = "<<endl<<mLeftP<<endl;

	Mat mRightRotation = R;
	Mat mRightTranslation = T;
	Mat mRightRT = Mat(3, 4, CV_64FC1);//右相机RT矩阵
	hconcat(mRightRotation, mRightTranslation, mRightRT);
	Mat mRightIntrinsic = cameraMatrixR;
	Mat mRightP = mRightIntrinsic * mRightRT;
	cout<<"右相机P矩阵 = "<<endl<<mRightP<<endl;

	//最小二乘法A矩阵
	Mat A = Mat(4, 3, CV_64FC1);
	A.at<double>(0, 0) = uvLeft.x * mLeftP.at<double>(2, 0) - mLeftP.at<double>(0, 0);
	A.at<double>(0, 1) = uvLeft.x * mLeftP.at<double>(2, 1) - mLeftP.at<double>(0, 1);
	A.at<double>(0, 2) = uvLeft.x * mLeftP.at<double>(2, 2) - mLeftP.at<double>(0, 2);

	A.at<double>(1, 0) = uvLeft.y * mLeftP.at<double>(2, 0) - mLeftP.at<double>(1, 0);
	A.at<double>(1, 1) = uvLeft.y * mLeftP.at<double>(2, 1) - mLeftP.at<double>(1, 1);
	A.at<double>(1, 2) = uvLeft.y * mLeftP.at<double>(2, 2) - mLeftP.at<double>(1, 2);

	A.at<double>(2, 0) = uvRight.x * mRightP.at<double>(2, 0) - mRightP.at<double>(0, 0);
	A.at<double>(2, 1) = uvRight.x * mRightP.at<double>(2, 1) - mRightP.at<double>(0, 1);
	A.at<double>(2, 2) = uvRight.x * mRightP.at<double>(2, 2) - mRightP.at<double>(0, 2);

	A.at<double>(3, 0) = uvRight.y * mRightP.at<double>(2, 0) - mRightP.at<double>(1, 0);
	A.at<double>(3, 1) = uvRight.y * mRightP.at<double>(2, 1) - mRightP.at<double>(1, 1);
	A.at<double>(3, 2) = uvRight.y * mRightP.at<double>(2, 2) - mRightP.at<double>(1, 2);

	//最小二乘法B矩阵
	Mat B = Mat(4, 1, CV_64FC1);
	B.at<double>(0, 0) = mLeftP.at<double>(0, 3) - uvLeft.x * mLeftP.at<double>(2, 3);
	B.at<double>(1, 0) = mLeftP.at<double>(1, 3) - uvLeft.y * mLeftP.at<double>(2, 3);
	B.at<double>(2, 0) = mRightP.at<double>(0, 3) - uvRight.x * mRightP.at<double>(2, 3);
	B.at<double>(3, 0) = mRightP.at<double>(1, 3) - uvRight.y * mRightP.at<double>(2, 3);

	Mat XYZ = Mat(3, 1, CV_64FC1);
	//采用SVD最小二乘法求解XYZ
	solve(A, B, XYZ, DECOMP_SVD);

	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;

	//世界坐标系中坐标
	Point3f world;
	world.x = XYZ.at<double>(0, 0);
	world.y = XYZ.at<double>(1, 0);
	world.z = XYZ.at<double>(2, 0);

	return world;
}
Mat disp2Depth(Mat disp)
{
	
	float fx = 943.96;
	float baseline = 102.34; //基线距离40mm

	Mat depth(disp.rows, disp.cols, CV_16S);  //深度图
	//视差图转深度图
	for (int row = 0; row < depth.rows; row++)
	{
		for (int col = 0; col < depth.cols; col++)
		{
			short d = disp.at<short>(row,col);
			//cout << d << " ";
			if (d <= 0)
			{
				//cout << depth.at<short>(row, col) << " ";
				continue;
			}
			depth.at<short>(row,col) = fx * baseline / d;
			//cout << depth.at<short>(row, col) << " ";
		}
	}
	/*
	for (int row = 0; row < depth.rows; row++)
	{
		for (int col = 0; col < depth.cols; col++)
		{
			cout << depth.at<short>(row,col) << " ";
		}
	}
	*/
	return depth;
}


void onMouse(int envent, int x, int y, int, void*userdata)
{
	Mat pospic;
	pospic = *(Mat*)userdata;
	string label("speedbump");
	if (selectObject)
	{
		selection.x = MIN(x, origin.x);
		selection.y = MIN(y, origin.y);
		selection.width = abs(x - origin.x);
		selection.height = abs(y - origin.y);

	}
	switch (envent)
	{
	case EVENT_LBUTTONDOWN:
		origin = Point(x, y);
		selection = Rect(x, y, 0, 0);
		selectObject = true;
		cout << origin << "in the world coordinate is: " << pospic.at<short>(origin) << endl;
		manualselectdepth = pospic.at<short>(origin);
		final_distance = 943 * 102.0 / manualselectdepth;
		label += to_string(final_distance) + " mm";
		labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		cout << "the manual select distance  z is " << 943 * 102.0 / manualselectdepth << endl;

		rectangle(rotsrc, Point(speedbox.tl().x, top - round(1.5*labelSize.height)), Point(speedbox.tl().x + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
		//putText(rotsrc, to_string(distance) + " mm", Point(speedbox.tl().x, speedbox.tl().y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
		putText(rotsrc, label, speedbox.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 178, 50), 1);
		imshow("rotated_rectiftysrc", rotsrc);

		break;
	case EVENT_LBUTTONUP:
		selectObject = false;
		if (selection.width > 0 && selection.height > 0)
			break;

	case EVENT_RBUTTONDOWN:
		Point pos = Point(x, y);
		cout << "pos" << pos.x << "  " << pos.y << endl;

	}
	
	


}



float measureDistance(Rect objectbox, Mat depthimg,Mat disparity)
{
	//vector<Mat> depthxyz;
	//split(depthimg, depthxyz);
	//imshow("depth", depthxyz[2]);

	short res_distance = -1;  //最终返回结果，当返回负值时，可表明测距失败
	//首先取识别结果边框的80%为有效视差区域
	int new_x0 = objectbox.tl().x + objectbox.width * 0.2 / 2;
	int new_y0 = objectbox.tl().y + objectbox.height * 0.2 / 2;
	int new_width = objectbox.width * 0.8;
	int new_height= objectbox.height * 0.8;
	Rect volid_objectbox(new_x0, new_y0, new_width, new_height);

	//在计算每个回环区域时，先存储相邻两个矩形的点集，再对此点集求差集即可
	vector<Rect> whole_rects;
	vector<Mat> rect_mats;
	vector<Mat> annular_mats;
	for (int i = 0; i != 5; ++i)
	{
		Rect rect;
		int rect_x0 = volid_objectbox.tl().x + volid_objectbox.width / 2 / 5 * i;
		int rect_y0 = volid_objectbox.tl().y + volid_objectbox.height / 2 / 5 * i;
		int rect_width = volid_objectbox.width / 5 * (5-i);
		int rect_height = volid_objectbox.height / 5 * (5 - i);
		rect = Rect(rect_x0, rect_y0, rect_width, rect_height);
		whole_rects.push_back(rect);

		
		//调试显示
		//ROI区域深度图
		//Mat rect_mat(depthxyz[2],rect);
		Mat rect_mat(depthimg, rect);
		Mat rect_indepence;
		rect_mat.copyTo(rect_indepence);
		rect_mats.push_back(rect_indepence);
		string window_name = "rect_" + to_string(i);
		//imshow(window_name, rect_mat);
		cout << window_name << rect << endl;
	}


	vector<Mat> annulars;
	for (int i = 0; i != rect_mats.size()-1; ++i)
	{
		//求各区域深度图差集
		//小图像在大图像中的矩形坐标位置
		
		int rect_x0 = 0 + whole_rects[0].width / 2 / 5;
		int rect_y0 = 0 + whole_rects[0].height / 2 / 5;
		int rect_width = whole_rects[i + 1].width;
		int rect_height = whole_rects[i + 1].height;
		Rect rect_roi(rect_x0, rect_y0, rect_width, rect_height);
		Mat roi_rect = rect_mats[i](rect_roi);
		roi_rect.setTo(0);
		string window_name = "annular_roi" + to_string(i);
		//imshow(window_name, rect_mats[i]);
		annulars.push_back(rect_mats[i]);

	}
	annulars.push_back(rect_mats[rect_mats.size() - 1]);
	
	//对于每一环形图像,求其均值和方差
	vector<double> us;
	vector<double> sigmas;
	vector<vector<short>> valid_valueall;
	for (int i = 0; i != annulars.size(); ++i)
	{
		vector<short> valid_value;
		for (int row = 0; row != annulars[i].rows; ++row)
		{
			for (int col = 0; col != annulars[i].cols; ++col)
			{
				if (annulars[i].at<short>(row, col) > 0)
				{
					valid_value.push_back(annulars[i].at<short>(row, col));
				}
			}
		}
		valid_valueall.push_back(valid_value);
	}
	for (int i = 0; i != valid_valueall.size(); ++i)
	{
		double sum = std::accumulate(std::begin(valid_valueall[i]), std::end(valid_valueall[i]), 0.0);
		double mean = sum / valid_valueall[i].size(); //均值  

		double accum = 0.0;
		std::for_each(std::begin(valid_valueall[i]), std::end(valid_valueall[i]), [&](const double d) {
			accum += (d - mean)*(d - mean);
		});

		double stdev = sqrt(accum / (valid_valueall[i].size() - 1)); //方差  
		us.push_back(mean);
		sigmas.push_back(stdev);
	}
	short distance_temp = 0.0;
	vector<float> percent{ 0.35,0.30,0.20,0.10,0.05 };
	for (int i = 0; i != valid_valueall.size(); ++i)
	{
		int minPosition = min_element(sigmas.begin(), sigmas.end()) - sigmas.begin();
		distance_temp += us[minPosition] * percent[i];
		sigmas.erase(sigmas.begin() + minPosition);
	}
	cout << "stable distace = " << distance_temp << endl;
	


	/*
	
	//取图像的差集
	vector<vector<Point>> annulars;
	for (int i = 0; i != 4; ++i)
	{
		vector<Point> outer_rect;
		for (int colpos = 0; colpos <= whole_rects[i].height; ++colpos)
		{
			for (int rowpos = 0; rowpos <= whole_rects[i].width; ++rowpos)
			{
				//cout << Point(whole_rects[i].tl().x + rowpos, whole_rects[i].tl().y + colpos) <<  " ";
				outer_rect.push_back(Point(whole_rects[i].tl().x + rowpos, whole_rects[i].tl().y + colpos));
			}
		}

		vector<Point> inner_rect;
		for (int colpos = 0; colpos <= whole_rects[i+1].height; ++colpos)
		{
			for (int rowpos = 0; rowpos <= whole_rects[i+1].width; ++rowpos)
			{
				inner_rect.push_back(Point(whole_rects[i+1].tl().x + rowpos, whole_rects[i+1].tl().y + colpos));
			}
		}
		vector<Point> annular;
		//std::set_difference(outer_rect.begin(), outer_rect.end(), inner_rect.begin(), inner_rect.end(), back_inserter(annular));
		for (auto beg = outer_rect.begin(); beg != outer_rect.end(); ++beg)
		{
			if (find(inner_rect.begin(), inner_rect.end(), *beg) == inner_rect.end())
			{
				annular.push_back(*beg);
			}
		}
		annulars.push_back(annular);
		annular.clear();

		
	}
	//将最内部矩形像素点加入
	vector<Point> inner_rect;
	for (int colpos = 0; colpos != whole_rects[whole_rects.size()-1].height; ++colpos)
	{
		for (int rowpos = 0; rowpos != whole_rects[whole_rects.size() - 1].width; ++rowpos)
		{
			inner_rect.push_back(Point(whole_rects[whole_rects.size() - 1].tl().x + rowpos, whole_rects[whole_rects.size() - 1].tl().y + colpos));
		}
	}
	annulars.push_back(inner_rect);
	
	
	//求每一区域均值
	vector<float> us_area;
	for (int i = 0; i != annulars.size(); ++i)
	{
		float sum_area=0.0;
		float u_area;
		for (int j = 0; j != annulars[i].size(); ++j)
		{
			sum_area += depthimg.at<Vec3f>(annulars[i][j])[2];
		}
		u_area = sum_area / annulars[i].size();
		us_area.push_back(u_area);
	}
	float sum = std::accumulate(std::begin(us_area), std::end(us_area), 0.0);
	cout << "u_distance = " << sum/us_area.size();

	*/
	res_distance = distance_temp;
	return res_distance;
}


//减速带识别所需
Mat preprocess(Mat src)
{

	Mat dst;
	/*
	Mat AINDANE_dst;
	AINDANE(src, AINDANE_dst);
	imshow("AINDANEenhance", AINDANE_dst);

	cv::Mat IAGCWD_dst;
	IAGCWD(src, IAGCWD_dst);
	imshow("IAGCWDenhance", IAGCWD_dst);
	*/

	//medianBlur(src, dst,5);
	//cvtColor(AINDANE_dst, dst, COLOR_BGR2GRAY);
	imshow("original image", src);
	//引导滤波
	/*
	Mat guided_dst;
	Mat src_gray;
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	//Mat p = dst;

	//dst = work(src);

	int r = 3;
	double eps = 0.1 * 0.1;
	eps *= 255 * 255;
	double time_guided = static_cast<double>(getTickCount());
	guided_dst = guidedFilter(src, src, r, eps);
	time_guided = ((double)getTickCount() - time_guided) / getTickFrequency();
	cout << "guided cost time " << time_guided << endl;
	//滤波会损失清晰度
	//imshow("adaptive median blur", dst);
	imshow("guided_imgae_filter", guided_dst);
	//双边滤波
	imwrite("guided_imgae_filter.jpg", guided_dst);
	*/
	/*
	Mat bilater_dst;
	double time_bilateral = static_cast<double>(getTickCount());
	bilateralFilter(src, bilater_dst, 20, 40, 10);
	time_bilateral = ((double)getTickCount() - time_bilateral) / getTickFrequency();

	cout << "bilateral cost time " << time_bilateral << endl;
	imshow("bilateral_filter", bilater_dst);
	imwrite("bilateral_filter.jpg", bilater_dst);
	*/

	Mat fast_bilateral2;
	src.copyTo(fast_bilateral2);
	double time_bilatera2 = static_cast<double>(getTickCount());
	CRBFilter(src.data, fast_bilateral2.data, src.cols, src.rows, src.cols *src.channels(), 0.05, 0.05);
	time_bilatera2 = ((double)getTickCount() - time_bilatera2) / getTickFrequency();
	cout << "bilateral cost time " << time_bilatera2 << endl;
	imshow("fast_bilateralfilter", fast_bilateral2);
	//imwrite("bilateral_filter2.jpg", fast_bilateral2);


	Mat AINDANE_dst;
	AINDANE(fast_bilateral2, AINDANE_dst);
	imshow("AINDANEenhance", AINDANE_dst);

	/*
	cvtColor(src, dst, COLOR_BGR2HSV);
	for (int row = 0; row < dst.rows; ++row)
	{
		for (int col = 0; col < dst.cols; ++col)
		{
			if ((dst.at<Vec3b>(row, col)[0] >= 11 && dst.at<Vec3b>(row, col)[0] <= 25) && (dst.at<Vec3b>(row, col)[1] >= 43 && dst.at<Vec3b>(row, col)[1] <= 255) && (dst.at<Vec3b>(row, col)[2] >= 46 && dst.at<Vec3b>(row, col)[2] <= 255))
			{
				//对于黄色进行增强
				int colorH = dst.at<Vec3b>(row, col)[0] + 2;
				int colorS = dst.at<Vec3b>(row, col)[1] * 1.5;
				int colorV = dst.at<Vec3b>(row, col)[2] * 1.5;

				if (colorH <= 25)
				{
					dst.at<Vec3b>(row, col)[0] = colorH;
				}
				else
				{
					dst.at<Vec3b>(row, col)[0] = 34;
				}

				if (colorS <= 255)
				{
					dst.at<Vec3b>(row, col)[1] = colorS;
				}
				else
					dst.at<Vec3b>(row, col)[1] = 255;


				if (colorV <= 255)
				{
					dst.at<Vec3b>(row, col)[2] = colorV;
				}
				else
					dst.at<Vec3b>(row, col)[2] = 255;


			}
		}

	}
	cvtColor(dst, dst, COLOR_HSV2BGR);
	imshow("enhance", dst);
	*/
	// mean shift 滤波与图像分割

	/*
	mean_src = guided_dst;
	resize(guided_dst, mean_src, Size(256, 256), 0, 0, 1);
	cvtColor(mean_src, mean_src, COLOR_RGB2Lab);
	// Initilize Mean Shift with spatial bandwith and color bandwith
	MeanShift MSProc(8, 16);
	// Filtering Process
	MSProc.MSFiltering(mean_src);
	// Segmentation Process include Filtering Process (Region Growing)
//	MSProc.MSSegmentation(Img);

	// Print the bandwith
	cout << "the Spatial Bandwith is " << MSProc.hs << endl;
	cout << "the Color Bandwith is " << MSProc.hr << endl;

	// Convert color from Lab to RGB
	cvtColor(mean_src, mean_src, COLOR_Lab2RGB);
	resize(mean_src, mean_src, Size(1279, 960), 0, 0);
	Mat diff;
	absdiff(src, mean_src,diff);
	imshow("diff", diff);
	// Show the result image
	namedWindow("MS Picture");
	imshow("MS Picture", mean_src);
	*/
	return AINDANE_dst;
}


Point2f triangle_top;
Point2f triangle_left, triangle_right;
Point2f getCrossPoint(Vec4i LineA, Vec4i LineB)
{
	double ka, kb;
	ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); //求出LineA斜率
	kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); //求出LineB斜率

	Point2f crossPoint;
	crossPoint.x = (ka*LineA[0] - LineA[1] - kb * LineB[0] + LineB[1]) / (ka - kb);
	crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka * LineB[1] - kb * LineA[1]) / (ka - kb);
	return crossPoint;
}

Mat draw_triangleroi(Mat& src, Point triangle_top, Point triangle_left, Point triangle_right)
{
	Mat dst;
	Mat roi = Mat::zeros(src.size(), CV_8U);
	vector<vector<Point>>contours;
	vector<Point>pts;
	pts.push_back(triangle_left);
	pts.push_back(triangle_top);
	pts.push_back(triangle_right);
	contours.push_back(pts);
	drawContours(roi, contours, 0, Scalar(255), -1);
	src.copyTo(dst, roi);
	return dst;

}

Mat roiTriangleExtraction(Mat& src, RotatedRect left, RotatedRect right)
{
	// 以车道线内侧边为搜索起始位置
	Point2f left_vertex[4];
	Point2f right_vertex[4];
	left.points(left_vertex);
	right.points(right_vertex);
	cout << endl;

	Vec4i lineLeft(left_vertex[1].x, left_vertex[1].y, left_vertex[2].x, left_vertex[2].y);
	Vec4i lineRight(right_vertex[3].x, right_vertex[3].y, right_vertex[2].x, right_vertex[2].y);
	cout << left_vertex[1].x << " " << left_vertex[1].y << " " << left_vertex[2].x << " " << left_vertex[2].y << endl;
	cout << right_vertex[3].x << " " << right_vertex[3].y << " " << right_vertex[2].x << " " << right_vertex[2].y << endl;

	triangle_left = Point(left_vertex[1].x, left_vertex[1].y);
	triangle_right = Point(right_vertex[3].x, right_vertex[3].y);

	triangle_top = getCrossPoint(lineLeft, lineRight);
	//cout << triangle_top << endl;
	Mat triangle_roi = draw_triangleroi(src, triangle_left, triangle_top, triangle_right);
	imshow("triangle_roi", triangle_roi);
	return triangle_roi;
}
vector<Mat> lsd_linefilter(Mat src, ntuple_list ntl, Point left_bottom, Point right_bottom)
{
	Mat all_lines;
	Mat tempimg;
	src.copyTo(tempimg);
	src.copyTo(all_lines);
	ntuple_list temp = new_ntuple_list(ntl->dim);
	vector<Point> vecp;
	vector<vector<Point>>vecps;
	Point pt1, pt2;
	float radian = atan2((right_bottom.y - left_bottom.y), (right_bottom.x - left_bottom.x));//弧度   该函数返回值范围是[-pi,pi]
	float angle = radian * 180 / 3.1415926;//角度
	//cout << "standard angle: " << angle << endl;
	for (int j = 0; j != ntl->size; ++j) //ntl的大小为图像中包含的线段数量
	{
		pt1.x = int(ntl->values[0 + j * ntl->dim]);
		pt1.y = int(ntl->values[1 + j * ntl->dim]);
		pt2.x = int(ntl->values[2 + j * ntl->dim]);
		pt2.y = int(ntl->values[3 + j * ntl->dim]);
		float curr_lineradian = atan2((pt2.y - pt1.y), (pt2.x - pt1.x));
		float curr_lineangle = curr_lineradian * 180 / 3.1415926;
		//cout << "curr_lineangle: " << curr_lineangle << endl;
		line(all_lines, pt1, pt2, Scalar(255), 2, 8);
		if (abs(curr_lineangle) < 10 || abs(abs(curr_lineangle) - 180) < 10) // 如果检测出的直线近似水平
		{
			//cout << "curr_lineangle: " << curr_lineangle << endl;
			vecp.clear();
			add_5tuple(temp, pt1.x, pt1.y, pt2.x, pt2.y, ntl->values[4 + j * ntl->dim]);
			vecp.push_back(pt1);
			vecp.push_back(pt2);
			vecps.push_back(vecp);
		}



	}
	imshow("all_lines", all_lines);
	//对temp按行进行排序
	sort(vecps.begin(), vecps.end(), [=](vector<Point>line1, vector<Point>line2) {return line1[0].y > line2[0].y; });

	// 对邻近行断开的直线进行连接
	for (int i = 0; i != vecps.size() - 1; ++i)
	{
		for (int j = i + 1; j != vecps.size(); ++j)
		{


		}
	}



	for (int j = 0; j != temp->size; ++j) //ntl的大小为图像中包含的线段数量
	{
		pt1.x = int(temp->values[0 + j * ntl->dim]);
		pt1.y = int(temp->values[1 + j * ntl->dim]);
		pt2.x = int(temp->values[2 + j * ntl->dim]);
		pt2.y = int(temp->values[3 + j * ntl->dim]);
		line(tempimg, pt1, pt2, Scalar(255), 2, 8);
	}
	Mat linesegmentimg(tempimg.size(), CV_8UC1, 0.0);
	for (int i = 0; i < tempimg.rows; i++)
	{
		for (int j = 0; j < tempimg.cols; j++)
		{
			if (tempimg.at<Vec3b>(i, j)[0] == 255)
			{
				linesegmentimg.at<uchar>(i, j) = 255;
			}

		}
	}

	imshow("connection", tempimg);
	Mat dilateElement = getStructuringElement(MORPH_RECT, Size(9, 9));
	dilate(linesegmentimg, linesegmentimg, dilateElement); // 膨胀函数
	imshow("binary", linesegmentimg);

	//对于缺损的减速带，减速带被分为两个区域，应对此两个区域进行合并，否则以下步骤无法外接旋转矩形


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(linesegmentimg, contours,
		hierarchy, RETR_EXTERNAL,
		CHAIN_APPROX_SIMPLE);
	//对区域进行合并
	vector<Rect>boundRect;
	vector<Point> boundRect_middle_pos;
	for (int i = 0; i != contours.size(); ++i)
	{
		boundRect.push_back(boundingRect(Mat(contours[i])));
		boundRect_middle_pos.push_back(Point(boundRect.back().x + boundRect.back().width / 2, boundRect.back().y + boundRect.back().height / 2));

		/*
		Mat temp(contours.at(i));
		Moments moment;//矩
		moment = moments(temp, false);
		Point pt1; // 重心
		if (moment.m00 != 0)//除数不能为0
		{
			pt1.x = cvRound(moment.m10 / moment.m00);//计算重心横坐标
			pt1.y = cvRound(moment.m01 / moment.m00);//计算重心纵坐标
		}
		cout << pt1 << endl;
		*/
	}
	vector<vector<Point>> allcontours;
	vector<Point> contourslist;
	for (int i = 0; i != contours.size(); ++i)
	{

		for (int j = 0; j != i; ++j)
		{
			if (abs(boundRect_middle_pos[i].y - boundRect_middle_pos[j].y) < 15)
			{
				contours[i].insert(contours[i].end(), contours[j].begin(), contours[j].end());
				break;
			}
		}

	}


	Mat res;
	src.copyTo(res);
	Mat res_rect;
	src.copyTo(res_rect);
	drawContours(res, contours,      //画出轮廓
		-1, // draw all contours
		Scalar(255, 255, 0), // in black
		2); // with a thickness of 2
	imshow("possibleregion", res);
	Mat roipos = Mat::zeros(src.size(), CV_8U);
	Mat result_speedimg;
	vector<Point>point;
	vector<vector<Point>> roicountour;


	if (!contours.empty())
	{
		for (size_t i = 0; i < contours.size(); ++i)
		{
			//====conditions for removing contours====//

			float contour_area = contourArea(contours[i]);

			//blob size should not be less than lower threshold
			if (contour_area > 5000)
			{
				RotatedRect rotated_rect = minAreaRect(contours[i]);
				Size2f sz = rotated_rect.size;
				float bounding_width = sz.width;
				float bounding_length = sz.height;
				if (bounding_length / bounding_width > 15 || bounding_width / bounding_length > 15)
				{
					Point2f vertex[4];
					rotated_rect.points(vertex);
					double aspect_ratio = 0.0;
					if (bounding_length > bounding_width)
					{
						cout << "aspect ratio = " << bounding_length / bounding_width << endl;
						aspect_ratio = bounding_length / bounding_width;
					}
					else
					{
						cout << "aspect ratio = " << bounding_width / bounding_length << endl;
						aspect_ratio = bounding_width / bounding_length;
					}

					for (int i = 0; i < 4; i++)
					{
						line(res_rect, vertex[i], vertex[(i + 1) % 4], Scalar(0, 0, 255), 1, 8);
						point.push_back(vertex[i]);
					}

				
					//输出宽高比
					//putText(res_rect, to_string(aspect_ratio), Point(rotated_rect.center.x, rotated_rect.center.y), FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255));

					//校正带旋转的矩形
					float angle = 0.0;

					if (sz.width <= sz.height)
					{
						angle = rotated_rect.angle + 90;
						int tm = sz.width;
						sz.width = sz.height;
						sz.height = tm;
						//swap(si.width, si.height);
					}
					else
					{
						angle = rotated_rect.angle;
					}

					//旋转矩形校正
					Mat rotmat = getRotationMatrix2D(rotated_rect.center, angle, 1);
					warpAffine(src, deal_img, rotmat, src.size(), INTER_CUBIC);
					rotmats.push_back(rotmat);
					imshow("deal_img", deal_img);
					Mat rRect;
					getRectSubPix(deal_img, sz, rotated_rect.center, rRect);
					//矩形区域
					Rect rect(rotated_rect.center.x - rRect.cols/2, rotated_rect.center.y - rRect.rows / 2,rRect.cols, rRect.rows);
					speedboxes.push_back(rect);
					imshow("rectArea", rRect);
					roi_rects.push_back(rRect);
					road_img.push_back(deal_img);
				}
			}
			if (!point.empty())
			{
				roicountour.push_back(point);
				point.clear();
			}

		}
		

		imshow("linerect", res_rect);
	}
	drawContours(roipos, roicountour, -1, Scalar::all(255), -1);
	src.copyTo(result_speedimg, roipos);
	//imshow("roipos", roipos);
	//imshow("result_speedimg", result_speedimg);
	cout << "speedboxes:" << speedboxes.size() << endl;
	cout << "roirects:" << roi_rects.size() << endl;
	return roi_rects;
}

vector<Mat> lsd_linedetection(Mat src)
{
	Mat gray_src, dst;
	src.copyTo(dst);
	cvtColor(src, gray_src, COLOR_RGB2GRAY);
	gray_src.convertTo(gray_src, CV_64FC1);

	int cols = gray_src.cols;
	int rows = gray_src.rows;

	image_double image = new_image_double(cols, rows);
	image->data = gray_src.ptr<double>(0);
	ntuple_list ntl;
	ntl = LineSegmentDetection(image, 0.4, 0.5, 2, 30, 2, 0.8, 1024, 255, NULL);
	Point pt1, pt2;
	for (int j = 0; j != ntl->size; ++j) //ntl的大小为图像中包含的线段数量
	{
		pt1.x = int(ntl->values[0 + j * ntl->dim]);
		pt1.y = int(ntl->values[1 + j * ntl->dim]);
		pt2.x = int(ntl->values[2 + j * ntl->dim]);
		pt2.y = int(ntl->values[3 + j * ntl->dim]);
		line(dst, pt1, pt2, Scalar(255, 0, 0), 1, 8);
	}
	vector<Mat> speedimg = lsd_linefilter(src, ntl, triangle_left, triangle_right);
	imshow("lsdLines", dst);
	return speedimg;
}


//直方图绘制函数，参数vector<int> nums 是灰度图片256级灰度的像素个数
void drawHist(vector<int> nums, string windowname)
{
	Mat hist = Mat::zeros(600, 800, CV_8UC3);
	auto Max = max_element(nums.begin(), nums.end());//max迭代器类型,最大数目
	putText(hist, "Histogram", Point(150, 100), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
	//*********绘制坐标系************//
	Point o = Point(100, 550);
	Point x = Point(700, 550);
	Point y = Point(100, 150);
	//x轴
	line(hist, o, x, Scalar(255, 255, 255), 2, 8, 0);
	//y轴
	line(hist, o, y, Scalar(255, 255, 255), 2, 8, 0);

	//********绘制灰度曲线***********//
	Point pts[256];
	//生成坐标点
	for (int i = 0; i < 256; i++)
	{
		pts[i].x = i * 2 + 100;
		pts[i].y = 550 - int(nums[i] * (300.0 / (*Max)));//归一化到[0, 300]
		//显示横坐标
		if ((i + 1) % 16 == 0)
		{
			string num = format("%d", i + 1);
			putText(hist, num, Point(pts[i].x, 570), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
		}
	}
	//绘制线
	for (int i = 1; i < 256; i++)
	{
		line(hist, pts[i - 1], pts[i], Scalar(0, 255, 0), 2);
	}
	//显示图像
	imshow(windowname, hist);
}
//计算直方图，统计各灰度级像素个数
void calHist(Mat& img, string windowname)
{
	Mat grey;
	if (img.channels() == 1)
	{
		grey = img;
	}
	else
		//先转为灰度图
		cvtColor(img, grey, COLOR_BGR2GRAY);
	//imshow("灰度图", grey);
	//计算各灰度级像素个数
	vector<int> nums(256);
	for (int i = 0; i < grey.rows; i++)
	{
		uchar* p = grey.ptr<uchar>(i);
		for (int j = 0; j < grey.cols; j++)
		{
			if (p[j] == 0)
				;
			else
				nums[p[j]]++;
		}
	}
	drawHist(nums, windowname);
}

void drawPredDistance(string classname, int left, int top, int right, int bottom,float distance,Mat& frame)
{

	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = classname + ": "+ to_string(distance);
	
	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}


int main(int argc, char* argv[])
{
	//printf("%s\n", avcodec_configuration());
	VideoCapture cap(1);
	cap.set(CAP_PROP_FOURCC, ('M','J','P','G'));
	cap.set(CAP_PROP_FPS, 60);
	cout << cap.get(CAP_PROP_FPS) << endl;
	cap.set(CAP_PROP_FRAME_WIDTH, 2560);
	cap.set(CAP_PROP_FRAME_HEIGHT, 960);
	Rect left_rect = Rect(0, 0, 1279, 960);
	Rect right_rect = Rect(1279, 0, 1279, 960);
	
	Mat frame;
	
	// 标定图像采集部分
	
	for (int i = 0; i <= 19; ++i)
	{
		
		/*
		waitKey(0);
		cap >> frame;
		Mat  left_img = Mat(frame, left_rect).clone();
		Mat  right_img = Mat(frame, right_rect).clone();

		imshow("frameL", left_img);
		imshow("frameR", right_img);

		waitKey(100);
		string picnameL = "C:/Users/54546/source/repos/dualCameraCalibration/dualCameraCalibration/Picture/" + to_string(i + 1) + "L" + ".jpg";
		string picnameR = "C:/Users/54546/source/repos/dualCameraCalibration/dualCameraCalibration/Picture/" + to_string(i + 1) + "R" + ".jpg";

		imwrite(picnameL, left_img);
		imwrite(picnameR, right_img);
		*/
	}
	
	//是否已完成标定，若完成标定则直接读取标定数据，若未完成标定则读取或拍摄标定棋盘格图像
	if (!hasCalibData)
	{
		Mat img;
		int goodFrameCount = 0;
		// 遍历棋盘格图像
		while (goodFrameCount < frameNumber)
		{
			char filename[100];
			/*读取左边的图像*/
			sprintf_s(filename, "C:/Users/54546/source/repos/dualCameraCalibration/dualCameraCalibration/Picture/%dL%s", goodFrameCount + 1, ".jpg");
			rgbImageL = imread(filename);
			cvtColor(rgbImageL, grayImageL, COLOR_BGR2GRAY);

			/*读取右边的图像*/
			sprintf_s(filename, "C:/Users/54546/source/repos/dualCameraCalibration/dualCameraCalibration/Picture/%dR%s", goodFrameCount + 1, ".jpg");
			rgbImageR = imread(filename);
			cvtColor(rgbImageR, grayImageR, COLOR_BGR2GRAY);

			bool isFindL, isFindR;

			isFindL = findChessboardCorners(rgbImageL, boardSize, cornerL);
			isFindR = findChessboardCorners(rgbImageR, boardSize, cornerR);
			if (isFindL == true && isFindR == true)  //如果两幅图像都找到了所有的角点 则说明这两幅图像是可行的  
			{
				/*
				Size(5,5) 搜索窗口的一半大小
				Size(-1,-1) 死区的一半尺寸
				TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 20, 0.1)迭代终止条件
				*/
				cornerSubPix(grayImageL, cornerL, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageL, boardSize, cornerL, isFindL);
				imshow("chessboardL", rgbImageL);
				imagePointL.push_back(cornerL);

				cornerSubPix(grayImageR, cornerR, Size(5, 5), Size(-1, -1), TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 20, 0.1));
				drawChessboardCorners(rgbImageR, boardSize, cornerR, isFindR);
				imshow("chessboardR", rgbImageR);
				imagePointR.push_back(cornerR);

				//string filename = "res\\image\\calibration";  
				//filename += goodFrameCount + ".jpg";  
				//cvSaveImage(filename.c_str(), &IplImage(rgbImage));       //把合格的图片保存起来  
				goodFrameCount++;
				cout << "The image" << goodFrameCount << " is good" << endl;
			}
			else
			{
				cout << "The image is bad please try again" << endl;
			}

			if (waitKey(10) == 'q')
			{
				break;
			}
		}

		/*
		计算实际的校正点的三维坐标
		根据实际标定格子的大小来设置
		*/
		calRealPoint(objRealPoint, boardWidth, boardHeight, frameNumber, squareSize);
		cout << "cal real successful" << endl;

		/*
		标定摄像头
		由于左右摄像机分别都经过了单目标定
		所以在此处选择flag = CALIB_USE_INTRINSIC_GUESS
		*/
		double rms = stereoCalibrate(objRealPoint, imagePointL, imagePointR,
			cameraMatrixL, distCoeffL,
			cameraMatrixR, distCoeffR,
			Size(imageWidth, imageHeight), R, T, E, F, CALIB_USE_INTRINSIC_GUESS,
			TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, 1e-5)); //需要注意，应该是版本的原因，该函数最                                                                                                                            后两个参数，我是调换过来后才显示不出错的

		cout << "Stereo Calibration done with RMS error = " << rms << endl;
		//对标定过的图像进行校正
		/*
		立体校正的时候需要两幅图像共面并且行对准 以使得立体匹配更加的可靠
		使得两幅图像共面的方法就是把两个摄像头的图像投影到一个公共成像面上，这样每幅图像从本图像平面投影到公共图像平面都需要一个旋转矩阵R
		stereoRectify 这个函数计算的就是从图像平面投影到公共成像平面的旋转矩阵Rl,Rr。 Rl,Rr即为左右相机平面行对准的校正旋转矩阵。
		左相机经过Rl旋转，右相机经过Rr旋转之后，两幅图像就已经共面并且行对准了。
		其中Pl,Pr为两个相机的投影矩阵，其作用是将3D点的坐标转换到图像的2D点的坐标:P*[X Y Z 1]' =[x y w]
		Q矩阵为重投影矩阵，即矩阵Q可以把2维平面(图像平面)上的点投影到3维空间的点:Q*[x y d 1] = [X Y Z W]。其中d为左右两幅图像的视差
		*/

		stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q,
			CALIB_ZERO_DISPARITY, -1, imageSize, &validROIL, &validROIR);
		/*
		根据stereoRectify 计算出来的R 和 P 来计算图像的映射表 mapx,mapy
		mapx,mapy这两个映射表接下来可以给remap()函数调用，来校正图像，使得两幅图像共面并且行对准
		ininUndistortRectifyMap()的参数newCameraMatrix就是校正后的摄像机矩阵。在openCV里面，校正后的计算机矩阵Mrect是跟投影矩阵P一起返回的。
		所以我们在这里传入投影矩阵P，此函数可以从投影矩阵P中读出校正后的摄像机矩阵
		*/
		//摄像机校正映射
		std::cout << validROIL << endl;
		std::cout << validROIR << endl;
		initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_32FC1, mapLx, mapLy);
		initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);
		outputCameraParam();
	}
	else
	{
		inputCameraParm();
	}
	

	

	//单张读取
	
	/*
	while (1)
	{
		cap >> frame;

		//Mat rectifyImageL, rectifyImageR;
		Mat  left_img = Mat(frame, left_rect).clone();
		Mat  right_img = Mat(frame, right_rect).clone();
		//rectifyImageL = imread("./Picture/1L.jpg");
		//rectifyImageR = imread("./Picture/1R.jpg");
		Mat rectifyImageL(left_img);
		Mat rectifyImageR(right_img);
		//cvtColor(grayImageL, rectifyImageL, COLOR_GRAY2BGR);
		//cvtColor(grayImageR, rectifyImageR, COLOR_GRAY2BGR);


		imshow("Rectify Before", rectifyImageL);
		cout << "按Q1退出 ..." << endl;

		
		//经过remap之后，左右相机的图像已经共面并且行对准了
		

		remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
		remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);

		cout << "按Q2退出 ..." << endl;

		imshow("rectifyImageL", rectifyImageL2);
		imshow("rectifyImageR", rectifyImageR2);

		setMouseCallback("disparity", onMouse, 0);
		stereo_match(0, 0);
		//stereo_SGBM_match(0, 0);
		//stereo_match1(0, 0);
		waitKey(0);

	}
	*/
	for(int numberpic=1; numberpic <=40;++numberpic)
	{
		string path = "./firsttest/";
		Mat rectifyImageL, rectifyImageR;
		rectifyImageL = imread(path+ to_string(numberpic)+"_left.jpg");
		rectifyImageR = imread(path + to_string(numberpic) + "_right.jpg");

		remap(rectifyImageL, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
		remap(rectifyImageR, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);

		imwrite("rectifyL.jpg", rectifyImageL2);
		imwrite("rectifyR.jpg", rectifyImageR2);
		//objectDetection(rectifyImageL2,);

		/*保存并输出数据*/

		//打开摄像头连续读取测试
		Mat testsgm_left = imread("1.png", 0);
		Mat testsgm_right = imread("2.png", 0);

		vector<string> classes = loadModel(modelWeightsDefaultMain, modelConfigurationDefaultMain, classesFileDefaultMain);
		while (false)
		{
			clock_t startTime, endTime;
			startTime = clock();//计时开始

			cap >> frame;
			Mat  left_img = Mat(frame, left_rect).clone();
			Mat  right_img = Mat(frame, right_rect).clone();
			remap(left_img, rectifyImageL2, mapLx, mapLy, INTER_LINEAR);
			remap(right_img, rectifyImageR2, mapRx, mapRy, INTER_LINEAR);

			endTime = clock();//计时结束
			cout << "The readimg time is: " << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;

			vector<int> indices;
			vector<int> classIds;
			vector<Rect> boxes;

			clock_t yolostartTime, yoloendTime;
			yolostartTime = clock();//计时开始
			objectDetection(rectifyImageL2, indices, classIds, boxes);

			yoloendTime = clock();//计时结束
			cout << "The yolo run time is: " << (double)(yoloendTime - yolostartTime) / CLOCKS_PER_SEC << "s" << endl;

			clock_t sgmstartTime, sgmendTime;
			sgmstartTime = clock();//计时开始
			cvtColor(rectifyImageL2, rectifyImageL2, COLOR_BGR2GRAY);
			cvtColor(rectifyImageR2, rectifyImageR2, COLOR_BGR2GRAY);

			Mat dip16s = SGMcuda(rectifyImageL2, rectifyImageR2);
			//imshow("disp16s", dip16s);
			
			//cout << dip8u << endl;
			//对于视差图较黑区域测距效果差

			//SGMcuda(testsgm_left, testsgm_right);
			//返回视差图，再根据重投影矩阵进行测距
			Mat distance3D;
			reprojectImageTo3D(dip16s, distance3D, Q, true);
			distance3D *= 16;
			Mat depthimage(dip16s.rows,dip16s.cols, CV_16S);
			depthimage = disp2Depth(dip16s);
			/*
			for (int i = 0; i != depthimage.rows; ++i)
			{
				for (int j = 0; j != depthimage.cols; ++j)
				{
					cout << depthimage.at<short>(i, j) << " ";
				}
			}
			*/
			setMouseCallback("SGMfilterdisparity", onMouse, &dip16s);

			//对yolo识别出的物体进行测距
			for (size_t i = 0; i < indices.size(); ++i)
			{
				int idx = indices[i];
				Rect box = boxes[idx];
				int classid = classIds[idx];

				cout << classes[classid] << " " << box << endl;


				int centerX = box.tl().x + box.width / 2;
				int centerY = box.tl().y + box.height / 2;
				Point center = Point(centerX, centerY);
				//中间点距离
				cout << "the middel pos distance  z is " << depthimage.ptr<short>(centerX)[centerY] << endl;
				
				//区域内最小值点的视差及对应深度
				Mat rectdepth(depthimage, box);
				float distance = 99999.99;
				for (int i = 0;i!=rectdepth.rows; ++i)
				{
					for (int j = 0; j != rectdepth.cols; ++j)
					{
						if (rectdepth.at<short>(i, j) > 100)
						{
							if(distance > rectdepth.at<short>(i, j))
								distance = rectdepth.at<short>(i, j);
						}
						else
							continue;
					}
				}

				
				cout << "the min distance z is " << distance << endl;

				
				
				float distance_measure = measureDistance(box, depthimage, dip16s);
				putText(left_img, classes[classid]+ " " + to_string(distance) + " mm", Point(box.tl().x, box.tl().y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);

				drawPredDistance(classes[classid], box.x, box.y, box.x + box.width, box.y + box.height, distance_measure, left_img);
			}
			//imshow("distance_img", dip8u);


			imshow("distance measure", left_img);


			sgmendTime = clock();//计时结束
			cout << "The sgm run time is: " << (double)(sgmendTime - sgmstartTime) / CLOCKS_PER_SEC << "s" << endl;

			/*
			bmcuda = cuda::createStereoBM(64, 19);
			cuda::GpuMat d_left, d_right;
			d_left.upload(rectifyImageL2);
			d_right.upload(rectifyImageR2);

			clock_t BMcudastartTime, BMcudaendTime;
			BMcudastartTime = clock();//计时开始
			cuda::GpuMat d_disp(rectifyImageL2.size(), CV_8U);
			bmcuda->compute(d_right, d_left, d_disp);

			Mat dispbmcuda(rectifyImageL2.size(), CV_8U);
			d_disp.download(dispbmcuda);

			BMcudaendTime = clock();//计时结束
			cout << "The bm cuda run time is: " << (double)(BMcudaendTime - BMcudastartTime) / CLOCKS_PER_SEC << "s" << endl;

			imshow("cudabmdisparity", dispbmcuda);
			*/

			//stereo_SGBM_match(0, 0);
			//setMouseCallback("depth", onMouse, 0);
			//setMouseCallback("SGMfilterdisparity", onMouse,0);
			
			waitKey(0);

		}



		/*
		把校正结果显示出来
		把左右两幅图像显示到同一个画面上
		这里只显示了最后一副图像的校正结果。并没有把所有的图像都显示出来
		*/
		Mat canvas;
		double sf;
		int w, h;
		sf = 600. / MAX(imageSize.width, imageSize.height);
		w = cvRound(imageSize.width * sf);
		h = cvRound(imageSize.height * sf);
		canvas.create(h, w * 2, CV_8UC3);

		/*左图像画到画布上*/
		Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
		resize(rectifyImageL2, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);  //把图像缩放到跟canvasPart一样大小  
		cvtColor(rectifyImageL2, rectifyImageL2, COLOR_BGR2GRAY);

		/*
		Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域
			cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
		*/
		//保留边缘畸变区域，使得图像大小一致
		Rect vroiL(0, 0,                //获得被截取的区域    
			cvRound(rectifyImageL2.cols*sf), cvRound(rectifyImageL2.rows*sf));
		rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  

		cout << "Painted ImageL" << endl;

		/*右图像画到画布上*/
		canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
		resize(rectifyImageR2, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
		cvtColor(rectifyImageR2, rectifyImageR2, COLOR_BGR2GRAY);

		/*
		Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
			cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
		*/
		Rect vroiR(0, 0,                //获得被截取的区域    
			cvRound(rectifyImageR2.cols*sf), cvRound(rectifyImageR2.rows*sf));
		rectangle(canvasPart, vroiR, Scalar(0, 255, 0), 3, 8);

		cout << "Painted ImageR" << endl;

		/*画上对应的线条*/
		for (int i = 0; i < canvas.rows; i += 16)
			line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);

		imshow("rectified", canvas);

		//测试图像本身已经过校正，所以不需要本代码中的remap
		//SGM要求图像为灰度图
		Mat rectifyImageL_gray, rectifyImageR_gray;
		cvtColor(rectifyImageL, rectifyImageL_gray, COLOR_BGR2GRAY);
		cvtColor(rectifyImageR, rectifyImageR_gray, COLOR_BGR2GRAY);


		clock_t SGMcpustartTime, SGMcpuendTime;
		SGMcpustartTime = clock();//计时开始

		Mat dip16s = SGMcuda(rectifyImageL2, rectifyImageR2);
		//SGMcuda(testsgm_left, testsgm_right);
		SGMcpuendTime = clock();//计时结束
		cout << "The sgm Gpu run time is: " << (double)(SGMcpuendTime - SGMcpustartTime) / CLOCKS_PER_SEC << "s" << endl;
		Mat depthimage(dip16s.rows, dip16s.cols, CV_16S);
		depthimage = disp2Depth(dip16s);
		Mat dip8u;
		dip16s.convertTo(dip8u, CV_8U);
		//视差图
		namedWindow("disparity", WINDOW_AUTOSIZE);
		/*
		createTrackbar("BlockSize:\n", "disparity", &blockSize, 8, stereo_match);
		createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
		createTrackbar("uniquenessRation:\n", "disparity", &uniquenessRation, 50, stereo_match);
		*/
		setMouseCallback("SGMfilterdisparity", onMouse, &dip16s);
		//setMouseCallback("rectifyImageL", onMouse, 0);
		//setMouseCallback("rectifyImageR", onMouse, 0);
		//setMouseCallback("disparity", onMouse, 0);

		//stereo_SGBM_match(0, 0);

		/*
		clock_t BMcpustartTime, BMcpuendTime;
		BMcpustartTime = clock();//计时开始
		stereo_match(0, 0);
		BMcpuendTime = clock();//计时结束
		cout << "The bm cpu run time is: " << (double)(BMcpuendTime - BMcpustartTime) / CLOCKS_PER_SEC << "s" << endl;

		//使用不同视差计算方法，鼠标点击后计算出的距离不一样啊！
		//Point3f real = uv2xyz(Point2f(251, 535), Point2f(249, 546));
		//cout << real.x << " " << real.y << "  " << real.z << endl;

		//cuda版本的BM算法


		cuda::GpuMat d_left, d_right;
		d_left.upload(rectifyImageL2);
		d_right.upload(rectifyImageR2);
		bmcuda = cuda::createStereoBM(64,19);
		bmcuda->setBlockSize(19);
		bmcuda->setPreFilterCap(31);
		bmcuda->setMinDisparity(0);
		bmcuda->setTextureThreshold(5);
		bmcuda->setUniquenessRatio(16);
		bmcuda->setSpeckleWindowSize(100);
		bmcuda->setSpeckleRange(32);
		bmcuda->setDisp12MaxDiff(-1);

		clock_t BMcudastartTime, BMcudaendTime;
		BMcudastartTime = clock();//计时开始
		cuda::GpuMat d_disp(rectifyImageL2.size(), CV_8U);
		bmcuda->compute(d_right, d_left, d_disp);


		Mat dispbmcuda(rectifyImageL2.size(), CV_8U);
		d_disp.download(dispbmcuda);

		BMcudaendTime = clock();//计时结束
		cout << "The bm cuda run time is: " << (double)(BMcudaendTime - BMcudastartTime) / CLOCKS_PER_SEC << "s" << endl;

		imshow("cudabmdisparity", dispbmcuda);
		*/
		/*
		//cpu-SGBM
		Mat SGBMmat = Match_SGBM(rectifyImageL2, rectifyImageR2);
		imshow("sgbmdisp", SGBMmat);
		*/

		//cuda-BP
		/*
		int ndisp = 128;
		int iter = 5;
		int level = 5;

		cuda::GpuMat d_left_bp, d_right_bp;
		cuda::GpuMat d_disp_bp(rectifyImageL2.size(), CV_32F);
		cuda::GpuMat d_disp_bp_8u;
		d_left_bp.upload(rectifyImageL2);
		d_right_bp.upload(rectifyImageR2);
		Ptr<cuda::StereoBeliefPropagation> sp = cuda::createStereoBeliefPropagation(64,10);
		//sp->estimateRecommendedParams(d_left_bp.cols, d_left_bp.rows, ndisp, iter, level);
		//sp->setNumDisparities(ndisp);
		//sp->setNumIters(iter);
		//sp->setNumLevels(level);
		sp->compute(d_left_bp, d_right_bp, d_disp_bp);
		Mat dispbmcuda_bp(rectifyImageL2.size(), CV_8U);
		d_disp_bp.convertTo(d_disp_bp_8u, CV_8U, 255. / ndisp);
		d_disp_bp_8u.download(dispbmcuda_bp);
		imshow("bp disparity", dispbmcuda_bp);
		*/

		//opencv_contrib版本sgm


		//添加减速带识别
		Mat preprocess_img = preprocess(rectifyImageL);
		Mat gray_img;
		cvtColor(preprocess_img, gray_img, COLOR_BGR2GRAY);
		vector<RotatedRect> left_rightLanepos;
		LaneDetect detect(gray_img, preprocess_img);
		left_rightLanepos = detect.edgeExtrction();
		Mat triangle_area = roiTriangleExtraction(preprocess_img, left_rightLanepos[0], left_rightLanepos[1]);


		circle(rectifyImageL, triangle_top, 12, Scalar(0, 0, 255), 1, 8);
		imshow("circle", rectifyImageL);
		Rect rect;

		vector<Mat> speedimg = lsd_linedetection(triangle_area);
		if (speedboxes.size() > 0)
		{
			//应对可能的减速带图像进一步筛选
		//从而进行缺损检测
			int i = 0;
			int max_mean = 0;
			int max_i = -1;
			for (; i != speedimg.size(); ++i)
			{
				//计算灰度图均值和方差
				Mat gray_speedimg, mat_mean, mat_var;
				cvtColor(speedimg[i], gray_speedimg, COLOR_BGR2GRAY);
				calHist(gray_speedimg, string("gray_speedimg"));
				meanStdDev(gray_speedimg, mat_mean, mat_var);
				double mean = mat_mean.at<double>(0, 0);
				double var = mat_var.at<double>(0, 0);
				if (mean > max_mean)
				{
					//均值最大者为可能感兴趣区域集合中的减速带图像
					max_mean = mean;
					max_i = i;
				}

			}
			string windowname = "speedroi";
			imshow(windowname, speedimg[max_i]);
			Mat curr_speedimg = speedimg[max_i];
			imshow("only choose speedimg", curr_speedimg);




			//对视差图和原图进行旋转
			Mat rotmat = rotmats[max_i];
			
			warpAffine(dip8u, rotdip8u, rotmat, dip8u.size(), INTER_CUBIC);
			warpAffine(rectifyImageL, rotsrc, rotmat, rectifyImageL.size(), INTER_CUBIC);

			//找到旋转后的减速带矩形框
			speedbox = speedboxes[max_i];
			speedboxes.clear();
			roi_rects.clear();
			Mat distance3D;
			reprojectImageTo3D(dip16s, distance3D, Q, true);

			int centerX = speedbox.tl().x + speedbox.width / 2;
			int centerY = speedbox.tl().y + speedbox.height / 2;
			Point center = Point(centerX, centerY);
			cout << "the distance  z is " << distance3D.at<Vec3f>(center) << endl;

			//float distance = 99999.99;
			//Mat rectdepth(depthimage, speedbox);
			//distance = measureDistance(speedbox, rectdepth, dip16s);
			string label("speedbump");
			
			
			label += to_string(final_distance) + "mm";
			labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
			rectangle(rotsrc, speedbox.tl(), speedbox.br(), Scalar(255, 178, 50), 3);
			top = speedbox.tl().y;
			top = max(top, labelSize.height);
			rectangle(rotsrc, Point(speedbox.tl().x, top - round(1.5*labelSize.height)), Point(speedbox.tl().x + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
			//putText(rotsrc, to_string(distance) + " mm", Point(speedbox.tl().x, speedbox.tl().y), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
			putText(rotsrc, label, speedbox.tl(), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255, 178, 50), 1);
			imshow("rotated_dips", rotdip8u);
			imshow("rotated_rectiftysrc", rotsrc);

		}
		

		waitKey(0);
		//system("pause");  
	}
	
	return 0;
}