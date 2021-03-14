#pragma execution_character_set("utf-8")

#include <iostream>
#include <opencv2/opencv.hpp>
#include "LaneDetecter.h"
#include "lsd.h"
#include "header.h"
//#include "EDLib.h"
#include <algorithm>
#include <opencv2/ximgproc.hpp>
#include "guidedfilter.h"
#include "MeanShift.h"
#include "header.h"
#include"Segmentation.h"
#include"AutoFitGamma.h"
#include "Util.h"
#include "image_enhancement.h"
#include "fastbilateralfilter.h"
#include "fastbilateral2.h"
using namespace std;
using namespace cv;
vector<Mat>roi_rects;
vector<Mat>road_img;


int textureDifference2(Vec4b p1, Vec4b p2)//current and seed texture distance metric for final merge
{
	//interchanged 1 and 2 return statements
	if (abs(p1[0] - p2[0]) < 60 && abs(p1[1] - p2[1]) < 30 && abs(p1[2] - p2[2]) < 60)// && abs(p1[3]-p2[3])<30)
		return 1;
	else
		return 0;

}
int farTextureDifference2(Vec4b p1, Vec4b p2)//local texture distance metric for final merge
{
	if (abs(p1[0] - p2[0]) < 10 && abs(p1[1] - p2[1]) < 5 && abs(p1[2] - p2[2]) < 10)// && abs(p1[3]-p2[3])<15)
		return 0;
	else
		return 0;

}

bool similar2(Vec3b now, Vec3b actual)//current and seed color distance metric for final merge
{
	//20 60 60
	//35 60 60
	if (abs(now[0] - actual[0]) < 3)// && abs(now[1]-actual[1])<60 && abs(now[2]-actual[2])<80)
		return 1;
	return 0;
}

bool farPtSimilar2(Vec3b next, Vec3b curr)//local color distance metric for final merge
{
	//2 5 5
	if (abs(next[0] - curr[0]) < 1)// && abs(next[1]-curr[1])<5 && abs(next[2]-curr[2])<7)
		return 0;
	return 0;
}



int ifSimilar(Vec4b nextTexture, Vec4b seedTexture, Vec4b currTexture,
	Vec3b nextColor, Vec3b seedColor, Vec3b currColor,
	int nextSegment, int seedSegment, int currSegment,
	int nextTotalPixels, int seedTotalPixels, int currTotalPixels)//distance metric score for final merge(change binary score to continuous valued)
{
	if (nextSegment == 0 && nextTexture[1] >= 1)
	{
		return textureDifference2(nextTexture, seedTexture);
	}
	else if (nextSegment != 0 && nextTexture[1] < 2)
	{
		return similar2(nextColor, seedColor);
	}
	else if (nextSegment != 0 && nextTexture[1] >= 1)
	{
		int cSim = 0, tSim = 0;
		if (similar2(nextColor, seedColor) || farPtSimilar2(nextColor, currColor))
			cSim = 1;
		if (textureDifference2(nextTexture, seedTexture) || farTextureDifference(nextTexture, currTexture))
			tSim = 1;

		if (tSim && cSim)
		{
			return 1;
		}
		else if (!tSim && cSim)
		{
			float rat = nextTotalPixels / currTotalPixels;
			if (0.85 <= rat && rat <= 1.5 && (nextTotalPixels > 50 || currTotalPixels > 50))//check absolute size with threshold also
			{
				//cout<<"fa\n";
				return 0;
			}
			else
			{
				//cout<<"tr\n";
				return 3;//return 1;
			}
		}
		else if (tSim && !cSim)
		{
			float rat = nextTotalPixels / currTotalPixels;
			if (0.60 <= rat && rat <= 1.5)//&& (nextTotalPixels > 50 || currTotalPixels > 50))//check absolute size with threshold also
			{
				return 1;
			}
			else
			{
				return 0;
			}
		}
		else
		{
			return 0;
		}

	}
	else
	{
		return 1;//consider merge
	}
}

Mat regionMerge(Mat image, Mat texture, Mat col, Mat segm, vector<int>pixelsInArea, vector<Vec3b>avgColBGR, int winSize)
{
	//merge the regions considering both texture and color segmentation
	Mat combined(col.rows, col.cols, CV_8UC3, 0.0);
	int jump = winSize / 2;//windowSize/2 in generate texture
	namedWindow("segfinal", 2);


	//trial blur
	//medianBlur(image, image, 3);
	medianBlur(segm, segm, 7);
	//medianBlur(texture, texture, 3);
	///
	/*namedWindow("1",2);
	namedWindow("2",2);
	namedWindow("3",2);
	imshow("1", image);
	imshow("2", segm);
	imshow("3", texture);*/
	///


	Point seed, curr, next;
	Mat marker(texture.rows, texture.cols, CV_8UC1, 0.0);
	queue<Point> Q;
	int segNo = 0;

	short dy, dx;

	for (int i = winSize; i < col.cols - winSize; i += jump)
	{
		for (int j = winSize; j < col.rows - winSize; j += jump)
		{
			seed.x = i / jump;
			seed.y = j / jump;

			//cout<<"seed : "<<seed.x<<" "<<seed.y<<" "<<endl;

			Vec4b seedTexture = texture.at<Vec4b>(seed.y, seed.x);//removed -1
			Vec3b seedColor = segm.at<Vec3b>(seed.y*jump, seed.x*jump);//check if col!=0
			int seedSegment = (int)col.at<uchar>(seed.y*jump, seed.x*jump);
			int seedTotalPixels = pixelsInArea[seedSegment - 1];

			if ((seedSegment == 0 && seedTexture[1] < 4) || marker.at<uchar>(seed.y, seed.x))
			{
				//cout<<"here:\n";
				continue;
			}

			segNo++;
			Q.push(seed);
			marker.at<uchar>(seed.y, seed.x) = segNo;

			while (!Q.empty())
			{
				curr = Q.front();
				Q.pop();

				Vec4b currTexture = texture.at<Vec4b>(curr.y, curr.x);
				Vec3b currColor = segm.at<Vec3b>(curr.y*jump, curr.x*jump);
				int currSegment = (int)col.at<uchar>(curr.y*jump, curr.x*jump);
				int currTotalPixels = pixelsInArea[currSegment - 1];


				for (int p = -jump; p <= jump; p += jump)
				{
					for (int q = -jump; q <= jump; q += jump)
					{
						next.x = curr.x + p / jump;
						next.y = curr.y + q / jump;
						if (0 <= next.x && next.x < texture.cols && 0 <= next.y && next.y < texture.rows
							&& marker.at<uchar>(next.y, next.x) == 0)
						{


							//cout<<next.y<< " "<<next.x<<endl;
							Vec4b nextTexture = texture.at<Vec4b>(next.y, next.x);
							Vec3b nextColor = segm.at<Vec3b>(next.y*jump, next.x*jump);
							int nextSegment = (int)col.at<uchar>(next.y*jump, next.x*jump);
							int nextTotalPixels = pixelsInArea[nextSegment - 1];

							//check if not zero marker then proceed
							int choice = ifSimilar(nextTexture, seedTexture, currTexture,
								nextColor, seedColor, currColor,
								nextSegment, seedSegment, currSegment,
								nextTotalPixels, seedTotalPixels, currTotalPixels);

							if (choice)//then merge both and color with updated color
							{
								//cout<<"push : "<<next.y<<" "<<next.x<<" "<<segNo<<endl;
								Q.push(next);//push the next pixel
								marker.at<uchar>(next.y, next.x) = 2; //mark the next pixel
								//color the block
								//check for " <= "
								if (choice == 3)
								{
									dx = -1 + (rand() % 3);
									dy = -1 + (rand() % 3);
									//cout<<"3";
								}

								for (int g = -jump; g <= jump; g++)
								{
									for (int h = -jump; h <= jump; h++)
									{
										if (0 <= (next.y*jump + h) && (next.y*jump + h) < combined.rows
											&& 0 <= (next.x*jump + g) && (next.x*jump + g) < combined.cols)
										{
											if (choice == 3)
											{
												if ((next.y + dy) < combined.rows && (next.x + dx) < combined.cols
													&& (next.y + dy) >= 0 && (next.x + dx) >= 0)
													combined.at<Vec3b>(next.y*jump + h, next.x*jump + g) = combined.at<Vec3b>(next.y*jump + dy, next.x*jump + dx);
												else
													combined.at<Vec3b>(next.y*jump + h, next.x*jump + g) = image.at<Vec3b>(seed.y*jump, seed.x*jump);
												//cout<<next.y*jump+h<<","<< next.x*jump+g<<",,"<<next.y + dy<<","<<next.x+ dx<<endl;
											}
											else
												combined.at<Vec3b>(next.y*jump + h, next.x*jump + g) = image.at<Vec3b>(seed.y*jump, seed.x*jump);//combined.at<Vec3b>(j+q, i+p);

										}
									}
								}
							}
						}

					}
				}



			}
			//imshow("segfinal",combined);
			//waitKey(2);
		}
	}

	///imshow("box",image);
	imshow("segfinal", combined);
	//resize(combined, combined, Size(320, 480));
	//namedWindow("hehe", 2);
	medianBlur(combined, combined, 11);
	//imshow("hehe", combined);
	waitKey(0);
	//imwrite("finalSegment.jpg",combined);
	return combined;
}


void crop(Mat combined, Mat original)
{
	//namedWindow("Smoothcombined", 2);
	//imshow("Smoothcombined", combined);

	namedWindow("finalll", 2);
	///Mat original=combined;
	imshow("try", original);

	int ROWS = combined.rows;
	int COLS = combined.cols;

	Mat marker(ROWS, COLS, CV_8UC1, 0.0);

	Point seed, curr, next;

	int segNo = 0;

	short dy, dx;

	int minx, maxx, miny, maxy;

	queue<Point> Q;
	cvtColor(combined, combined, COLOR_BGR2HSV);
	///convert the image to HSV space
	for (int i = 0; i < COLS; ++i)
	{
		for (int j = 0; j < ROWS; ++j)
		{
			seed.x = i;
			seed.y = j;
			minx = maxx = seed.x;
			miny = maxy = seed.y;
			//cout<<"seed : "<<seed.x<<" "<<seed.y<<" "<<endl;

			Vec3b seedColor = combined.at<Vec3b>(seed.y, seed.x);//check if col!=0
			//cout<<seed.x<<","<<seed.y<<endl;
			//cout<<marker.at<uchar>(seed.y, seed.x)<<" ";
			if (marker.at<uchar>(seed.y, seed.x) != 0)
			{
				//cout<<"here:\n";
				continue;
			}

			segNo++;
			Q.push(seed);
			marker.at<uchar>(seed.y, seed.x) = segNo;

			while (!Q.empty())
			{
				curr = Q.front();
				Q.pop();

				Vec3b currColor = combined.at<Vec3b>(curr.y, curr.x);

				for (int p = -1; p <= 1; ++p)
				{
					for (int q = -1; q <= 1; ++q)
					{
						next.x = curr.x + p;
						next.y = curr.y + q;
						if (0 <= next.x && next.x < COLS && 0 <= next.y && next.y < ROWS
							&& marker.at<uchar>(next.y, next.x) == 0)
						{


							//cout<<next.y<< " "<<next.x<<endl;
							Vec3b nextColor = combined.at<Vec3b>(next.y, next.x);

							if (abs(nextColor[0] - seedColor[0]) <= 2)//choice)//then merge both and color with updated color
							{
								//cout<<"push : "<<next.y<<" "<<next.x<<" "<<segNo<<endl;
								Q.push(next);//push the next pixel
								marker.at<uchar>(next.y, next.x) = 2;//segNo; //mark the next pixel
								//color the block
								//check for " <= "
								if (miny > next.y)
									miny = next.y;
								if (maxy < next.y)
									maxy = next.y;
								if (maxx < next.x)
									maxx = next.x;
								if (minx > next.x)
									minx = next.x;
							}
						}//cout<<maxx<<","<<minx<<","<<maxy<<","<<miny<<endl;
					}
				}
			}
			if (abs((maxx - minx)*(maxy - miny)) > 10000 && abs(maxx - minx) > 50 && abs(maxy - miny) > 50)
			{
				rectangle(original, Point(minx, maxy), Point(maxy, miny), Scalar(128, 255, 0), 5);
				waitKey(0);
				imshow("finalll", original);
			}
		}
	}
	imshow("finalll", original);
	waitKey(0);
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

Mat draw_triangleroi(Mat& src,Point triangle_top, Point triangle_left, Point triangle_right)
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

Mat roiTriangleExtraction(Mat& src,RotatedRect left, RotatedRect right)
{
	// 以车道线内侧边为搜索起始位置
	Point2f left_vertex[4];
	Point2f right_vertex[4];
	left.points(left_vertex);
	right.points(right_vertex);
	cout << endl;
	
	Vec4i lineLeft(left_vertex[1].x, left_vertex[1].y, left_vertex[2].x, left_vertex[2].y);
	Vec4i lineRight(right_vertex[3].x, right_vertex[3].y, right_vertex[2].x, right_vertex[2].y);
	cout <<  left_vertex[1].x << " " << left_vertex[1].y << " " << left_vertex[2].x  <<" "<< left_vertex[2].y << endl;
	cout << right_vertex[3].x << " " << right_vertex[3].y << " " << right_vertex[2].x << " " << right_vertex[2].y << endl;

	triangle_left = Point(left_vertex[1].x, left_vertex[1].y);
	triangle_right = Point(right_vertex[3].x, right_vertex[3].y);

	triangle_top = getCrossPoint(lineLeft, lineRight);
	//cout << triangle_top << endl;
	Mat triangle_roi = draw_triangleroi(src, triangle_left, triangle_top, triangle_right);
	imshow("triangle_roi", triangle_roi);
	return triangle_roi;
}
vector<Mat> lsd_linefilter(Mat src,ntuple_list ntl, Point left_bottom, Point right_bottom)
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
		if(abs(curr_lineangle) < 10 || abs(abs(curr_lineangle) - 180) < 10) // 如果检测出的直线近似水平
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
	for (int i = 0; i != vecps.size()-1; ++i)
	{
		for (int j = i+1; j != vecps.size(); ++j)
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
	Mat linesegmentimg(tempimg.size(),CV_8UC1,0.0);
	for(int i = 0; i < tempimg.rows; i++)
	{
		for(int j = 0; j < tempimg.cols; j++)
		{
			if (tempimg.at<Vec3b>(i,j)[0] == 255)
			{
				linesegmentimg.at<uchar>(i,j) = 255;
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
		boundRect_middle_pos.push_back(Point(boundRect.back().x + boundRect.back().width/2, boundRect.back().y + boundRect.back().height / 2));
		
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
			if (abs(boundRect_middle_pos[i].y - boundRect_middle_pos[j].y) <  15)
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
		Scalar(255,255,0), // in black
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
					putText(res_rect, to_string(aspect_ratio), Point(rotated_rect.center.x, rotated_rect.center.y), FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255));

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
					Mat rotmat = getRotationMatrix2D(rotated_rect.center, angle, 1);
					Mat deal_img;
					warpAffine(src, deal_img, rotmat, src.size(), INTER_CUBIC);
					imshow("deal_img", deal_img);
					Mat rRect;
					getRectSubPix(deal_img, sz, rotated_rect.center, rRect);
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
	drawContours(roipos, roicountour,-1,Scalar::all(255),-1);
	src.copyTo(result_speedimg, roipos);
	//imshow("roipos", roipos);
	//imshow("result_speedimg", result_speedimg);

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
	ntl = LineSegmentDetection(image,0.4,0.5,2,30,2,0.8,1024,255,NULL);
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

void EDCOLOR(Mat src)
{
	//EDColor testEDColor = EDColor(src, 36, 4, 1.5, true); //last parameter for validation
	//imshow("Color Edge Image - PRESS ANY KEY TO QUIT", testEDColor.getEdgeImage());
	//EDLines colorLine = EDLines(src);
	//imshow("Color Line", colorLine.getLineImage());
	//std::cout << "Number of line segments: " << colorLine.getLinesNo() << std::endl;
}

void findContours(Mat src)
{

	int min_area = 100;
	int max_area = 500;

	Mat gray_src;
	cvtColor(src, gray_src, COLOR_BGR2GRAY);
	Mat edge;
	Canny(gray_src, edge, 50, 150, 3);
	//threshold(gray_src, gray_src, 100, 255,THRESH_BINARY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edge, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat result(src.size(), CV_8U, Scalar(0));
	drawContours(result, contours,      //画出轮廓
		-1, // draw all contours
		Scalar(255), // in black
		2); // with a thickness of 2
	imshow("lunluo", result);

	

}

/*
uchar adaptiveProcess(const Mat &im, int row, int col, int kernelSize, int maxSize)
{
	vector <uchar> pixels;
	for (int a = -kernelSize / 2; a <= kernelSize / 2; a++) {
		for (int b = -kernelSize / 2; b <= kernelSize / 2; b++) {
			pixels.push_back(im.at<uchar>(row + a, col + b));
		}
	}
	sort(pixels.begin(), pixels.end());
	auto min = pixels[0];
	auto max = pixels[kernelSize * kernelSize - 1];
	auto med = pixels[kernelSize * kernelSize / 2];
	auto zxy = im.at<uchar>(row, col);
	if (med > min && med < max) {
		if (zxy > min && zxy < max) {
			return zxy;
		}
		else {
			return med;
		}
	}
	else {
		kernelSize += 2;
		if (kernelSize <= maxSize)
			return adaptiveProcess(im, row, col, kernelSize, maxSize);
		else
			return med;
	}
}
*/
// 自适应中值滤波
/*
Mat work(Mat src) {
	Mat dst;
	int minSize = 3; //滤波器窗口的起始大小
	int maxSize = 7; //滤波器窗口的最大尺寸
	copyMakeBorder(src, dst, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, BORDER_REFLECT);
	int rows = dst.rows;
	int cols = dst.cols;
	for (int j = maxSize / 2; j < rows - maxSize / 2; j++) {
		for (int i = maxSize / 2; i < cols * dst.channels() - maxSize / 2; i++) {
			dst.at<uchar>(j, i) = adaptiveProcess(dst, j, i, minSize, maxSize);
		}
	}
	return dst;
}
*/
//-------------------------------------------------------------
//作者：不用先生，2019.8.11
//自实现的导向滤波去噪算法
//guidedFilter.cpp
//-------------------------------------------------------------


//--------------------------------------------------------------
//函数名：my_guidedFilter_oneChannel
//函数功能：用于单通道图像（灰度图）的引导滤波函数；
//参数：Mat &srcImg：输入图像，为单通道图像；
//参数：Mat &guideImg：引导图像，为单通道图像，尺寸与输入图像一致；
//参数：Mat &dstImg：输出图像，尺寸、通道数与输入图像吻合；
//参数：const int rad：滤波器大小，应该保证为奇数，默认值为9；
//参数：const double eps ：防止a过大的正则化参数，
/*
bool my_guidedFilter_oneChannel(Mat &srcImg, Mat &guideImg, Mat &dstImg, const int rad = 9, const double eps = 0.01)
{
	//--------------确保输入参数正确-----------
	{
		if (!srcImg.data || srcImg.channels() != 1)
		{
			cout << "输入图像错误，请重新输入图像" << endl;
			return false;
		}

		if (!guideImg.data || guideImg.channels() != 1)
		{
			cout << "输入引导图像错误，请重新输入图像" << endl;
			return false;
		}

		if (guideImg.cols != srcImg.cols || guideImg.rows != srcImg.rows)
		{
			cout << "输入图像与引导图像尺寸不匹配，请重新确认" << endl;
			return false;
		}
		if (dstImg.cols != srcImg.cols || dstImg.rows != srcImg.rows || dstImg.channels() != 1)
		{
			cout << "参数输出图像错误，请重新确认" << endl;
			return false;
		}
		if (rad % 2 != 1)
		{
			cout << "参数“rad”应为奇数，请修改" << endl;
			return false;
		}

	}

	//--------------转换数值类型，并归一化-------------
	srcImg.convertTo(srcImg, CV_32FC1, 1.0 / 255.0);
	guideImg.convertTo(guideImg, CV_32FC1, 1.0 / 255.0);

	//--------------求引导图像和输入图像的均值图
	Mat mean_srcImg, mean_guideImg;
	boxFilter(srcImg, mean_srcImg, CV_32FC1, Size(rad, rad));
	boxFilter(guideImg, mean_guideImg, CV_32FC1, Size(rad, rad));

	Mat mean_guideImg_square, mean_guideImg_srcImg;
	boxFilter(guideImg.mul(guideImg), mean_guideImg_square, CV_32FC1, Size(rad, rad));
	boxFilter(guideImg.mul(srcImg), mean_guideImg_srcImg, CV_32FC1, Size(rad, rad));

	Mat var_guideImg = mean_guideImg_square - mean_guideImg.mul(mean_guideImg);
	Mat cov_guideImg_srcImg = mean_guideImg_srcImg - mean_guideImg.mul(mean_srcImg);

	Mat aImg = cov_guideImg_srcImg / (var_guideImg + eps);
	Mat bImg = mean_srcImg - aImg.mul(mean_guideImg);

	Mat mean_aImg, mean_bImg;
	boxFilter(aImg, mean_aImg, CV_32FC1, Size(rad, rad));
	boxFilter(bImg, mean_bImg, CV_32FC1, Size(rad, rad));

	dstImg = (mean_aImg.mul(guideImg) + mean_bImg);

	dstImg.convertTo(dstImg, CV_8UC1, 255);

	return true;
}
*/
//--------------------------------------------------------------
//函数名：my_guidedFilter_threeChannel
//函数功能：用于三通道图像（RGB彩色图）的引导滤波函数；
//参数：Mat &srcImg：输入图像，为三通道图像；
//参数：Mat &guideImg：引导图像，为三通道图像，尺寸与输入图像一致；
//参数：Mat &dstImg：输出图像，尺寸、通道数与输入图像吻合；
//参数：const int rad：滤波器大小，应该保证为奇数，默认值为9；
//参数：const double eps ：防止a过大的正则化参数，

/*
bool my_guidedFilter_threeChannel(Mat &srcImg, Mat &guideImg, Mat &dstImg, const int rad = 2, const double eps = 0.01)
{
	//----------------确保输入参数正确-------------
	{
		if (!srcImg.data || srcImg.channels() != 3)
		{
			cout << "输入图像错误，请重新输入图像" << endl;
			return false;
		}

		if (!guideImg.data || guideImg.channels() != 3)
		{
			cout << "输入引导图像错误，请重新输入图像" << endl;
			return false;
		}
		if (guideImg.cols != srcImg.cols || guideImg.rows != srcImg.rows)
		{
			cout << "输入图像与引导图像尺寸不匹配，请重新确认" << endl;
			return false;
		}
		if (rad % 2 != 1)
		{
			cout << "参数“rad”应为奇数，请修改" << endl;
			return false;
		}

	}

	vector<Mat> src_vec, guide_vec, dst_vec;
	split(srcImg, src_vec);
	split(guideImg, guide_vec);

	for (int i = 0; i < 3; i++)
	{
		Mat tempImg = Mat::zeros(srcImg.rows, srcImg.cols, CV_8UC1);
		my_guidedFilter_oneChannel(src_vec[i], guide_vec[i], tempImg, rad, eps);
		dst_vec.push_back(tempImg);
	}
	
	merge(dst_vec, dstImg);

	return true;
}
*/



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
void calHist(Mat& img,string windowname)
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

vector<int> basedYellowBlock(vector<Mat> blocks)
{
	//循环计算每一色块的均值，统计高于均值的像素数量与低于此均值的像素数量比
 	vector<float> ratios;
	
	for (int i = 0; i != blocks.size(); ++i)
	{
		int sum = 0;
		for (int row = 0; row != blocks[i].rows; ++row)
		{
			for (int col = 0; col != blocks[i].cols; ++col)
			{
				sum += blocks[i].at<uchar>(row, col);
			}
		}
		int average = sum / (blocks[i].rows * blocks[i].cols);

		
		//float ratio = float(A2) / A1;
		ratios.push_back(average);
	}
	//找到最大比值的下标，认为该位置大概率为黄色色块区域
	auto maxValue = max_element(ratios.begin(),ratios.end());
	int pos = maxValue - ratios.begin();
	//以此为基准，推测其他位置色块颜色
	vector<int> colors;
	colors.resize(ratios.size());
	colors[pos] = 1;
	int boolnext = 0;
	cout << "the" + to_string(pos) + "block color is" << colors[pos] << endl;

	for (int i = pos; i >= 0; --i)
	{
		if (!boolnext)
		{
			colors[i] = 1;
			boolnext = 1;
		}
		else
		{
			colors[i] = 0;
			boolnext = 0;
		}

	}
	boolnext = 0;
	for (int i = pos + 1; i != ratios.size(); ++i)
	{
		if (!boolnext)
		{
			colors[i] = 0;
			boolnext = 1;
		}
		else
		{
			colors[i] = 1;
			boolnext = 0;
		}

	}
	for (auto beg = colors.begin(); beg != colors.end(); ++beg)
	{
		cout << *beg << " ";
	}
	cout << endl;

	return colors;
}
vector<int> grad_mutation_detection(vector<int> row_grad)
{
	vector<int> exist_mutation;
	// 对当前色块梯度值进行检测
	// 使用Mann-Kendall突变点检测方法

	//正序检测
	vector<int> SK;
	vector<float> UFK;
	int s =0;
	vector<float>exp_value;
	vector<float>var_value;
	for (int i = 1; i != row_grad.size(); ++i)
	{
		for (int j = 0; j != i; ++j)
		{
			if (row_grad[i] > row_grad[j])
				s += 1;
			else
				s += 0;
		}
		SK.push_back(s);
		exp_value.push_back((i + 1) * (i + 2) / 4.0);
		var_value.push_back((i + 1) * i * (2 * (i + 1) + 5) / 72.0);
		UFK.push_back((SK.back() - exp_value.back()) / sqrt(var_value.back()));
	}

	//逆序检测
	vector<int> SK2;
	vector<float> UBK;
	vector<float> UBK2;

	int s2 = 0;
	vector<float>exp_value2;
	vector<float>var_value2;
	//对行梯度进行反转
	vector<int> row_grad_reverse = row_grad;
	reverse(row_grad_reverse.begin(), row_grad_reverse.end());

	for (int i = 1; i != row_grad_reverse.size(); ++i)
	{
		for (int j = 0; j != i; ++j)
		{
			if (row_grad_reverse[i] > row_grad_reverse[j])
				s2 += 1;
			else
				s2 += 0;
		}
		SK2.push_back(s2);
		exp_value2.push_back((i + 1) * (i + 2) / 4.0);
		var_value2.push_back((i + 1) * i * (2 * (i + 1) + 5) / 72.0);
		UBK.push_back((SK2.back() - exp_value2.back()) / sqrt(var_value2.back()));
		UBK2.push_back(-UBK.back());
	}
	vector<float>UBKT = UBK2;
	reverse(UBKT.begin(), UBKT.end());
	vector<float> diff;
	for (int i = 0; i != UFK.size(); ++i)
	{
		diff.push_back(UFK[i] - UBKT[i]);
	}
	vector<int>cross_pos;
	for (int k = 1; k != UBK.size(); ++k)
	{
		if ((diff[k - 1] * diff[k]) < 0)
		{
			cross_pos.push_back(k);
			cout << k << " ";
		}
			
	}
	cout << endl;
	return cross_pos;
}

int main()
{
	int beg_picnum = 1;
	int end_picnum = 24;

	char filename[100];
	
	for (int beg = beg_picnum; beg != end_picnum; ++beg)
	{
		sprintf_s(filename, "./jiansudai/%d_left.jpg",beg);
		Mat img = imread(filename);
		imshow("原图",img);
		
		Mat preprocess_img = preprocess(img);
		Mat gray_img;
		cvtColor(preprocess_img, gray_img, COLOR_BGR2GRAY);
		vector<RotatedRect> left_rightLanepos;
		LaneDetect detect(gray_img, preprocess_img);
		left_rightLanepos = detect.edgeExtrction();
		Mat triangle_area = roiTriangleExtraction(preprocess_img,left_rightLanepos[0], left_rightLanepos[1]);


		circle(img, triangle_top, 12, Scalar(0,0,255), 1, 8);
		imshow("circle", img);
		Rect rect;

		vector<Mat> speedimg = lsd_linedetection(triangle_area);
		//应对可能的减速带图像进一步筛选
		//从而进行缺损检测
		int i = 0;
		int max_mean = 0;
		int max_i = -1;
		for(;i != speedimg.size(); ++i)
		{
			//计算灰度图均值和方差
			Mat gray_speedimg, mat_mean,mat_var;
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
		/*
		//根据灰度图使用soble横向边缘检测，效果较差
		Mat gray_speed;
		cvtColor(speedimg[i], gray_speed, COLOR_BGR2GRAY);
		Mat sobel_speed;
		Sobel(gray_speed, sobel_speed, CV_8U, 1, 0, 3);
		imshow("sobel_speed", sobel_speed);
		*/

		/*
		//基于纹理和颜色的分割方法，难以运行
		hist(speedimg[i]);
		int winSize = 10;
		Mat combined = colSeg(speedimg[i], winSize);
		crop(combined, speedimg[i]);
		*/

		//Mean Shift滤波效果较差
		/*
		resize(speedimg[i], speedimg[i],Size(0,0),2,2);
		Mat mean_src = speedimg[i];
		//resize(speedimg[i], mean_src, Size(256, 256), 0, 0, 1);
		cvtColor(mean_src, mean_src, COLOR_RGB2Lab);
		// Initilize Mean Shift with spatial bandwith and color bandwith
		MeanShift MSProc(8, 16);
		// Filtering Process
		MSProc.MSFiltering(mean_src);
		imshow("small block", mean_src);
		*/

		//统计路面区域灰度直方图
		calHist(road_img[max_i], "road_hist");
		imshow("road_img", road_img[max_i]);
		//calHist(speedimg[max_i]);
		Mat gray_speed, grayspeed_copy;
		cvtColor(speedimg[max_i], gray_speed, COLOR_BGR2GRAY);
		gray_speed.copyTo(grayspeed_copy);
		imshow("gray_speed", gray_speed);
		for (int i = 0; i != gray_speed.rows; ++i)
		{
			for (int j = 0; j != gray_speed.cols; ++j)
			{
				if (gray_speed.at<uchar>(i, j) < 68 && gray_speed.at<uchar>(i, j) > 15)
				{
					gray_speed.at<uchar>(i, j) -= 10;
				}
				else if (gray_speed.at<uchar>(i, j) >= 80)
				{
					gray_speed.at<uchar>(i, j) += 40;
				}
			}
		}
		imshow("gray_enhance", gray_speed);
		//自定义模板计算梯度
		/*
		int rows = gray_speed.rows;
		Mat grad_template(rows,3,CV_8UC1);
		for (int i = 0; i < rows; ++i)
		{
			grad_template.at<uchar>(i, 0) = -2;
			grad_template.at<uchar>(i, 1) = 0;
			grad_template.at<uchar>(i, 2) = 2;

		}
		*/
		vector<Point> datas;
		vector<float> datas_y;
		vector<int> datas_x;

		for (int col = 1; col < gray_speed.cols - 1; ++col)
		{
			int sum1 = 0;
			int sum2 = 0;
			int sum3 = 0;

			for (int i = 0; i != gray_speed.rows; ++i)
			{
				sum1 += gray_speed.at<uchar>(i, col - 1);
				sum2 += gray_speed.at<uchar>(i, col);
				sum3 += gray_speed.at<uchar>(i, col + 1);

			}
			//cout << sum3 - sum1 << endl;
			datas.push_back(Point(col, abs(2*sum3 - 2*sum1)));
			datas_y.push_back(abs(2*sum3 - 2*sum1));
			datas_x.push_back(col);

		}


		cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC3);
		image.setTo(cv::Scalar(255, 255, 255));

		//绘制折线
		cv::polylines(image, datas, false, cv::Scalar(0, 0, 0), 1, 8, 0);
		imshow("data", image);


		// 按从大到小排序


		vector<Point> local_peaks;
		vector<float>local_peaks_y;
		vector<int> local_peaks_x;
		vector<int> index_out;
		Peaks::findPeaks(datas_y, index_out);

		cout << "Maxima found:" << endl;

		for (int i = 0; i < index_out.size(); ++i)
		{
			
			if (index_out[i] == datas_y.size()) //越界
			{
				cout << datas_y[index_out[i-1]] << " ";
				local_peaks.push_back(Point(index_out[i], datas_y[index_out[i - 1]]));
			}
				
			else
			{
				cout << datas_y[index_out[i]] << " ";
				local_peaks.push_back(Point(index_out[i], datas_y[index_out[i]]));
			}
				

		}

		//对邻近峰值进行剔除
		sort(local_peaks.begin(), local_peaks.end(), [=](const Point& p1, const Point& p2) {
			return p1.x < p2.x; });
		for (auto beg = local_peaks.begin(); beg != local_peaks.end() - 1;)
		{
			if ((beg + 1)->x - beg->x < 21)
			{
				if (beg->y < (beg + 1)->y)
				{
					beg = local_peaks.erase(beg);
					continue;
				}
				else
				{
					auto next = local_peaks.erase(beg + 1);
					beg = next - 1;
				}

			}
			else
			{
				++beg;
			}
		}


		cv::Mat image2 = image;
		//绘制局部峰值圆圈
		for (int i = 0; i < local_peaks.size(); ++i)
		{
			circle(image2, local_peaks[i], 5, Scalar(0, 0, 0), 1);
		}

		imshow("peaks", image2);

		//有时切割存在问题
		vector<Point> interval_points; //计算间隔的峰值点存储容器
		vector<int> interval; //相邻点间隔计算结果存储
		//对上述得到的峰值点进行排序
		sort(local_peaks.begin(), local_peaks.end(), [=](const Point& p1, const Point& p2) {
			return p1.y > p2.y;
		});

		int speed_block_interval = 0; //初始化减速带宽度值
		bool is_find_interval = false; //初始化循环判定条件

		while (!is_find_interval)
		{
			while (interval_points.size() < 3) //计算间隔至少需三个峰值点
			{
				interval_points.push_back(local_peaks.front()); //将峰值点压入待检测容器
				local_peaks.erase(local_peaks.begin());//同时删除此峰值点
			}
			//对峰值点按x从小到大排序
			sort(interval_points.begin(), interval_points.end(), [=](const Point p1, const Point p2) { return p1.x < p2.x; });
			for (int i = 0; i != interval_points.size() - 1; ++i)
			{
				const int nums = interval_points.size() - 1;
				interval.push_back(interval_points[i + 1].x - interval_points[i].x);
			}
			int sum_interval = 0; //可用间距累计值
			vector<int> select_interval; //可用间距值存储容器
			int record = 0; //可用间距值数量
			int i = 0;
			for (; i != interval.size() - 1; ++i)
			{
				//找到了两组等间距值，认为此间距为色块宽度大小
				if (record >= 2)
				{
					is_find_interval = true;
					break;
				}
				//相邻区域间隔一致时，对其进行存储
				if (abs(interval[i] - interval[i + 1]) <= 10)
				{
					record += 1;
					sum_interval += max(interval[i], interval[i + 1]);
					select_interval.push_back(max(interval[i], interval[i + 1]));
					continue;
				}
				//相邻间隔不一致时，遍历下一相近间隔
				else if (abs(interval[i] - interval[i + 1]) > 10)
				{
					continue;
				}
			}
			if (record == 2)
			{
				/*
				if (i >=8)
				{
					vector<int> interval_sort = interval;
					sort(interval_sort.begin(), interval_sort.end());
					int midlle_interval = interval_sort[(interval_sort.size()/2) + 1];
					speed_block_interval = midlle_interval;

				}
				*/
				//计算减速带色块宽度
				if (abs(select_interval[0] - select_interval[1]) < 10)
				{
					speed_block_interval = sum_interval / record;
					break;
				}
				//若遍历完仍无两组相近宽度值可用，则取其中值
				if (local_peaks.empty())
				{
					is_find_interval == true;
					sort(interval.begin(), interval.end());
					speed_block_interval = interval[interval.size() / 2];
					break;
				}
				//当待检测容器不足两组可用间距时，加入新的峰值点，并将该峰值点从原容器中删除
				else
				{
					interval_points.push_back(local_peaks.front());
					local_peaks.erase(local_peaks.begin());
					interval.clear();
					record = 0;
					sum_interval = 0;
					is_find_interval = false;
				}
			}
		
			else
			{
				if (local_peaks.empty())
				{
					is_find_interval == true;
					sort(interval.begin(), interval.end());
					speed_block_interval = interval[interval.size() / 2];
					break;
				}
				else
				{
					interval_points.push_back(local_peaks.front());
					local_peaks.erase(local_peaks.begin());
					interval.clear();
					record = 0;
					sum_interval = 0;
				}
				
			}


		}

		vector<Mat> blocks; // 存储每个区块
		//找到间隔后对图像进行分割
		cout << "split interval is " << speed_block_interval << endl;
		//int new_speed_interval = gray_speed.cols  / floor(gray_speed.cols/ speed_block_interval);
		int i_block = 0;
		for (; i_block <= (gray_speed.cols / speed_block_interval) - 1; ++i_block)
		{
			Mat block(grayspeed_copy, Rect(i_block*speed_block_interval, 0, speed_block_interval, gray_speed.rows));
			string block_pos = to_string(i_block+1) + "block";
			string block_hist = to_string(i_block) + "hist";

			namedWindow(block_pos, WINDOW_NORMAL);
			imshow(block_pos, block);
			blocks.push_back(block);
			calHist(block, block_hist);
		}
		if (i_block == (gray_speed.cols / speed_block_interval) && (gray_speed.cols % speed_block_interval) > speed_block_interval / 1.4)
		{
			Mat block(grayspeed_copy, Rect(i_block*speed_block_interval, 0, gray_speed.cols - (i_block)*speed_block_interval, gray_speed.rows));
			string block_pos = to_string(i_block+1) + "block";
			namedWindow(block_pos, WINDOW_NORMAL);
			imshow(block_pos, block);
			blocks.push_back(block);
			i_block = 0;
		}
		//以亮度均值最高为黄色块，依次推断其余色块颜色。
		vector<int>colors_seq = basedYellowBlock(blocks);
		vector<bool>defect; //记录是否缺失
		//判断是否缺损
		//计算每一行梯度
		for (int i = 0; i != blocks.size(); ++i)
		{
			vector<int> row_sums;
			vector<int> row_grad;
			for (int row = 0; row != blocks[i].rows; ++row)
			{
				int row_sum = 0;
				for (int col = 0; col != blocks[i].cols; ++col)
				{
					row_sum += blocks[i].at<uchar>(row, col);
				}
				row_sums.push_back(row_sum);
			}
			for (int i = 1; i != row_sums.size(); ++i)
			{
				row_grad.push_back(abs(row_sums[i] - row_sums[i - 1]));
				//cout << abs(row_sums[i] - row_sums[i - 1]) << " ";

			}
			//cout << endl;
			//对每幅图梯度进行绘制
			cv::Mat grad_image = cv::Mat::zeros(1280, 640, CV_8UC3);
			grad_image.setTo(cv::Scalar(100, 0, 0));
			vector<Point> points;
			cout << "img_grad:" << " ";
			for (int i = 0; i != row_grad.size(); ++i)
			{
				points.push_back(Point(i * 10, row_grad[i]));
				cout << row_grad[i] << " ";
			}
			cout << endl;
			cv::polylines(grad_image, points, false, cv::Scalar(0, 255, 0), 1, 8, 0);
			string grad_str = to_string(i) + "gradimg";
			imshow(grad_str, grad_image);

			//返回交点坐标，坐标表示突变开始位置
			vector<int>cross_pos = grad_mutation_detection(row_grad);

			if (cross_pos.size() >= 1)
			{
				defect.push_back(false);
				/*
				int first_pos = cross_pos.front();
				int last_pos = cross_pos.back();
				int front_area = blocks[i].rows / 3;
				int back_area = blocks[i].rows / 3 * 2;
				if (first_pos <= front_area || last_pos >= back_area)
					defect.push_back(false);
				else
					defect.push_back(true);
				*/
			}
			else
			{
				defect.push_back(true);
			}


		}
		//结果输出
		for (int i = 0; i != colors_seq.size(); ++i)
		{
			if (colors_seq[i] == 0)
			{
				// true为缺损
				if (defect[i])
					cout << "the" + to_string(i + 1) << "block color is black, not exist" << endl;
				else
					cout << "the" + to_string(i + 1) << "block color is black, is exist" << endl;

			}
			else
			{
				if (defect[i])
					cout << "the" + to_string(i + 1) << "block color is yellow, not exist" << endl;
				else
					cout << "the" + to_string(i + 1) << "block color is yellow, is exist" << endl;

			}
		}
		cout << endl;



	

		//hist(speedimg);
		//int winSize = 40;
		//pair<Mat, vector<pair<pair<Point, Vec3b> ,int > > >retVal=
		//Mat combined = colSeg(speedimg, winSize);
		//crop(combined, speedimg);
		
		/*
		resize(speedimg, speedimg, Size(256, 256), 0, 0, 1);
		cvtColor(speedimg, speedimg, COLOR_RGB2Lab);
		// Initilize Mean Shift with spatial bandwith and color bandwith
		MeanShift MSProc(8, 16);
		// Filtering Process
		MSProc.MSFiltering(speedimg);
		// Segmentation Process include Filtering Process (Region Growing)
	//	MSProc.MSSegmentation(Img);

		// Print the bandwith
		cout << "the Spatial Bandwith is " << MSProc.hs << endl;
		cout << "the Color Bandwith is " << MSProc.hr << endl;

		// Convert color from Lab to RGB
		cvtColor(speedimg, speedimg, COLOR_Lab2RGB);
		imshow("color enhance", speedimg);
		*/
		//int winSize = 20;
		//Mat combined = colSeg(img, winSize);
		//crop(combined, triangle_area);
		//cvtColor(img, img, COLOR_BGR2GRAY);
		//findContours(triangle_area);

		roi_rects.clear();
		road_img.clear();
		imshow("lanedetect",img);
		waitKey(0);
	}


	return 0;
}