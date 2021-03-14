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
	ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); //���LineAб��
	kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); //���LineBб��

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
	// �Գ������ڲ��Ϊ������ʼλ��
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
	float radian = atan2((right_bottom.y - left_bottom.y), (right_bottom.x - left_bottom.x));//����   �ú�������ֵ��Χ��[-pi,pi]
	float angle = radian * 180 / 3.1415926;//�Ƕ�
	//cout << "standard angle: " << angle << endl;
	for (int j = 0; j != ntl->size; ++j) //ntl�Ĵ�СΪͼ���а������߶�����
	{
		pt1.x = int(ntl->values[0 + j * ntl->dim]);
		pt1.y = int(ntl->values[1 + j * ntl->dim]);
		pt2.x = int(ntl->values[2 + j * ntl->dim]);
		pt2.y = int(ntl->values[3 + j * ntl->dim]);
		float curr_lineradian = atan2((pt2.y - pt1.y), (pt2.x - pt1.x));
		float curr_lineangle = curr_lineradian * 180 / 3.1415926;
		//cout << "curr_lineangle: " << curr_lineangle << endl;
		line(all_lines, pt1, pt2, Scalar(255), 2, 8);
		if(abs(curr_lineangle) < 10 || abs(abs(curr_lineangle) - 180) < 10) // ���������ֱ�߽���ˮƽ
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
	//��temp���н�������
	sort(vecps.begin(), vecps.end(), [=](vector<Point>line1, vector<Point>line2) {return line1[0].y > line2[0].y; });
	
	// ���ڽ��жϿ���ֱ�߽�������
	for (int i = 0; i != vecps.size()-1; ++i)
	{
		for (int j = i+1; j != vecps.size(); ++j)
		{
			

		}
	}

	

	for (int j = 0; j != temp->size; ++j) //ntl�Ĵ�СΪͼ���а������߶�����
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
	dilate(linesegmentimg, linesegmentimg, dilateElement); // ���ͺ���
	imshow("binary", linesegmentimg);

	//����ȱ��ļ��ٴ������ٴ�����Ϊ��������Ӧ�Դ�����������кϲ����������²����޷������ת����


	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(linesegmentimg, contours,
		hierarchy, RETR_EXTERNAL,
		CHAIN_APPROX_SIMPLE);
	//��������кϲ�
	vector<Rect>boundRect;
	vector<Point> boundRect_middle_pos;
	for (int i = 0; i != contours.size(); ++i)
	{
		boundRect.push_back(boundingRect(Mat(contours[i])));
		boundRect_middle_pos.push_back(Point(boundRect.back().x + boundRect.back().width/2, boundRect.back().y + boundRect.back().height / 2));
		
		/*
		Mat temp(contours.at(i));
		Moments moment;//��
		moment = moments(temp, false);
		Point pt1; // ����
		if (moment.m00 != 0)//��������Ϊ0
		{
			pt1.x = cvRound(moment.m10 / moment.m00);//�������ĺ�����
			pt1.y = cvRound(moment.m01 / moment.m00);//��������������
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
	drawContours(res, contours,      //��������
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
					//�����߱�
					putText(res_rect, to_string(aspect_ratio), Point(rotated_rect.center.x, rotated_rect.center.y), FONT_HERSHEY_COMPLEX, 2, Scalar(0, 0, 255));

					//У������ת�ľ���
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
	for (int j = 0; j != ntl->size; ++j) //ntl�Ĵ�СΪͼ���а������߶�����
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
	drawContours(result, contours,      //��������
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
// ����Ӧ��ֵ�˲�
/*
Mat work(Mat src) {
	Mat dst;
	int minSize = 3; //�˲������ڵ���ʼ��С
	int maxSize = 7; //�˲������ڵ����ߴ�
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
//���ߣ�����������2019.8.11
//��ʵ�ֵĵ����˲�ȥ���㷨
//guidedFilter.cpp
//-------------------------------------------------------------


//--------------------------------------------------------------
//��������my_guidedFilter_oneChannel
//�������ܣ����ڵ�ͨ��ͼ�񣨻Ҷ�ͼ���������˲�������
//������Mat &srcImg������ͼ��Ϊ��ͨ��ͼ��
//������Mat &guideImg������ͼ��Ϊ��ͨ��ͼ�񣬳ߴ�������ͼ��һ�£�
//������Mat &dstImg�����ͼ�񣬳ߴ硢ͨ����������ͼ���Ǻϣ�
//������const int rad���˲�����С��Ӧ�ñ�֤Ϊ������Ĭ��ֵΪ9��
//������const double eps ����ֹa��������򻯲�����
/*
bool my_guidedFilter_oneChannel(Mat &srcImg, Mat &guideImg, Mat &dstImg, const int rad = 9, const double eps = 0.01)
{
	//--------------ȷ�����������ȷ-----------
	{
		if (!srcImg.data || srcImg.channels() != 1)
		{
			cout << "����ͼ���������������ͼ��" << endl;
			return false;
		}

		if (!guideImg.data || guideImg.channels() != 1)
		{
			cout << "��������ͼ���������������ͼ��" << endl;
			return false;
		}

		if (guideImg.cols != srcImg.cols || guideImg.rows != srcImg.rows)
		{
			cout << "����ͼ��������ͼ��ߴ粻ƥ�䣬������ȷ��" << endl;
			return false;
		}
		if (dstImg.cols != srcImg.cols || dstImg.rows != srcImg.rows || dstImg.channels() != 1)
		{
			cout << "�������ͼ�����������ȷ��" << endl;
			return false;
		}
		if (rad % 2 != 1)
		{
			cout << "������rad��ӦΪ���������޸�" << endl;
			return false;
		}

	}

	//--------------ת����ֵ���ͣ�����һ��-------------
	srcImg.convertTo(srcImg, CV_32FC1, 1.0 / 255.0);
	guideImg.convertTo(guideImg, CV_32FC1, 1.0 / 255.0);

	//--------------������ͼ�������ͼ��ľ�ֵͼ
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
//��������my_guidedFilter_threeChannel
//�������ܣ�������ͨ��ͼ��RGB��ɫͼ���������˲�������
//������Mat &srcImg������ͼ��Ϊ��ͨ��ͼ��
//������Mat &guideImg������ͼ��Ϊ��ͨ��ͼ�񣬳ߴ�������ͼ��һ�£�
//������Mat &dstImg�����ͼ�񣬳ߴ硢ͨ����������ͼ���Ǻϣ�
//������const int rad���˲�����С��Ӧ�ñ�֤Ϊ������Ĭ��ֵΪ9��
//������const double eps ����ֹa��������򻯲�����

/*
bool my_guidedFilter_threeChannel(Mat &srcImg, Mat &guideImg, Mat &dstImg, const int rad = 2, const double eps = 0.01)
{
	//----------------ȷ�����������ȷ-------------
	{
		if (!srcImg.data || srcImg.channels() != 3)
		{
			cout << "����ͼ���������������ͼ��" << endl;
			return false;
		}

		if (!guideImg.data || guideImg.channels() != 3)
		{
			cout << "��������ͼ���������������ͼ��" << endl;
			return false;
		}
		if (guideImg.cols != srcImg.cols || guideImg.rows != srcImg.rows)
		{
			cout << "����ͼ��������ͼ��ߴ粻ƥ�䣬������ȷ��" << endl;
			return false;
		}
		if (rad % 2 != 1)
		{
			cout << "������rad��ӦΪ���������޸�" << endl;
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
	//�����˲�
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
	//�˲�����ʧ������
	//imshow("adaptive median blur", dst);
	imshow("guided_imgae_filter", guided_dst);
	//˫���˲�
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
				//���ڻ�ɫ������ǿ
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
	// mean shift �˲���ͼ��ָ�

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


//ֱ��ͼ���ƺ���������vector<int> nums �ǻҶ�ͼƬ256���Ҷȵ����ظ���
void drawHist(vector<int> nums, string windowname)
{
	Mat hist = Mat::zeros(600, 800, CV_8UC3);
	auto Max = max_element(nums.begin(), nums.end());//max����������,�����Ŀ
	putText(hist, "Histogram", Point(150, 100), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
	//*********��������ϵ************//
	Point o = Point(100, 550);
	Point x = Point(700, 550);
	Point y = Point(100, 150);
	//x��
	line(hist, o, x, Scalar(255, 255, 255), 2, 8, 0);
	//y��
	line(hist, o, y, Scalar(255, 255, 255), 2, 8, 0);

	//********���ƻҶ�����***********//
	Point pts[256];
	//���������
	for (int i = 0; i < 256; i++)
	{
		pts[i].x = i * 2 + 100;
		pts[i].y = 550 - int(nums[i] * (300.0 / (*Max)));//��һ����[0, 300]
		//��ʾ������
		if ((i + 1) % 16 == 0)
		{
			string num = format("%d", i + 1);
			putText(hist, num, Point(pts[i].x, 570), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
		}
	}
	//������
	for (int i = 1; i < 256; i++)
	{
		line(hist, pts[i - 1], pts[i], Scalar(0, 255, 0), 2);
	}
	//��ʾͼ��
	imshow(windowname, hist);
}
//����ֱ��ͼ��ͳ�Ƹ��Ҷȼ����ظ���
void calHist(Mat& img,string windowname)
{
	Mat grey;
	if (img.channels() == 1)
	{
		grey = img;
	}
	else
	//��תΪ�Ҷ�ͼ
		cvtColor(img, grey, COLOR_BGR2GRAY);
	//imshow("�Ҷ�ͼ", grey);
	//������Ҷȼ����ظ���
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
	//ѭ������ÿһɫ��ľ�ֵ��ͳ�Ƹ��ھ�ֵ��������������ڴ˾�ֵ������������
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
	//�ҵ�����ֵ���±꣬��Ϊ��λ�ô����Ϊ��ɫɫ������
	auto maxValue = max_element(ratios.begin(),ratios.end());
	int pos = maxValue - ratios.begin();
	//�Դ�Ϊ��׼���Ʋ�����λ��ɫ����ɫ
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
	// �Ե�ǰɫ���ݶ�ֵ���м��
	// ʹ��Mann-Kendallͻ����ⷽ��

	//������
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

	//������
	vector<int> SK2;
	vector<float> UBK;
	vector<float> UBK2;

	int s2 = 0;
	vector<float>exp_value2;
	vector<float>var_value2;
	//�����ݶȽ��з�ת
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
		imshow("ԭͼ",img);
		
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
		//Ӧ�Կ��ܵļ��ٴ�ͼ���һ��ɸѡ
		//�Ӷ�����ȱ����
		int i = 0;
		int max_mean = 0;
		int max_i = -1;
		for(;i != speedimg.size(); ++i)
		{
			//����Ҷ�ͼ��ֵ�ͷ���
			Mat gray_speedimg, mat_mean,mat_var;
			cvtColor(speedimg[i], gray_speedimg, COLOR_BGR2GRAY);
			calHist(gray_speedimg, string("gray_speedimg"));
			meanStdDev(gray_speedimg, mat_mean, mat_var);
			double mean = mat_mean.at<double>(0, 0);
			double var = mat_var.at<double>(0, 0);
			if (mean > max_mean)
			{
				//��ֵ�����Ϊ���ܸ���Ȥ���򼯺��еļ��ٴ�ͼ��
				max_mean = mean;
				max_i = i;
			}

		}
		string windowname = "speedroi";
		imshow(windowname, speedimg[max_i]);
		Mat curr_speedimg = speedimg[max_i];
		imshow("only choose speedimg", curr_speedimg);
		/*
		//���ݻҶ�ͼʹ��soble�����Ե��⣬Ч���ϲ�
		Mat gray_speed;
		cvtColor(speedimg[i], gray_speed, COLOR_BGR2GRAY);
		Mat sobel_speed;
		Sobel(gray_speed, sobel_speed, CV_8U, 1, 0, 3);
		imshow("sobel_speed", sobel_speed);
		*/

		/*
		//�����������ɫ�ķָ������������
		hist(speedimg[i]);
		int winSize = 10;
		Mat combined = colSeg(speedimg[i], winSize);
		crop(combined, speedimg[i]);
		*/

		//Mean Shift�˲�Ч���ϲ�
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

		//ͳ��·������Ҷ�ֱ��ͼ
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
		//�Զ���ģ������ݶ�
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

		//��������
		cv::polylines(image, datas, false, cv::Scalar(0, 0, 0), 1, 8, 0);
		imshow("data", image);


		// ���Ӵ�С����


		vector<Point> local_peaks;
		vector<float>local_peaks_y;
		vector<int> local_peaks_x;
		vector<int> index_out;
		Peaks::findPeaks(datas_y, index_out);

		cout << "Maxima found:" << endl;

		for (int i = 0; i < index_out.size(); ++i)
		{
			
			if (index_out[i] == datas_y.size()) //Խ��
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

		//���ڽ���ֵ�����޳�
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
		//���ƾֲ���ֵԲȦ
		for (int i = 0; i < local_peaks.size(); ++i)
		{
			circle(image2, local_peaks[i], 5, Scalar(0, 0, 0), 1);
		}

		imshow("peaks", image2);

		//��ʱ�и��������
		vector<Point> interval_points; //�������ķ�ֵ��洢����
		vector<int> interval; //���ڵ����������洢
		//�������õ��ķ�ֵ���������
		sort(local_peaks.begin(), local_peaks.end(), [=](const Point& p1, const Point& p2) {
			return p1.y > p2.y;
		});

		int speed_block_interval = 0; //��ʼ�����ٴ����ֵ
		bool is_find_interval = false; //��ʼ��ѭ���ж�����

		while (!is_find_interval)
		{
			while (interval_points.size() < 3) //������������������ֵ��
			{
				interval_points.push_back(local_peaks.front()); //����ֵ��ѹ����������
				local_peaks.erase(local_peaks.begin());//ͬʱɾ���˷�ֵ��
			}
			//�Է�ֵ�㰴x��С��������
			sort(interval_points.begin(), interval_points.end(), [=](const Point p1, const Point p2) { return p1.x < p2.x; });
			for (int i = 0; i != interval_points.size() - 1; ++i)
			{
				const int nums = interval_points.size() - 1;
				interval.push_back(interval_points[i + 1].x - interval_points[i].x);
			}
			int sum_interval = 0; //���ü���ۼ�ֵ
			vector<int> select_interval; //���ü��ֵ�洢����
			int record = 0; //���ü��ֵ����
			int i = 0;
			for (; i != interval.size() - 1; ++i)
			{
				//�ҵ�������ȼ��ֵ����Ϊ�˼��Ϊɫ���ȴ�С
				if (record >= 2)
				{
					is_find_interval = true;
					break;
				}
				//����������һ��ʱ��������д洢
				if (abs(interval[i] - interval[i + 1]) <= 10)
				{
					record += 1;
					sum_interval += max(interval[i], interval[i + 1]);
					select_interval.push_back(max(interval[i], interval[i + 1]));
					continue;
				}
				//���ڼ����һ��ʱ��������һ������
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
				//������ٴ�ɫ����
				if (abs(select_interval[0] - select_interval[1]) < 10)
				{
					speed_block_interval = sum_interval / record;
					break;
				}
				//����������������������ֵ���ã���ȡ����ֵ
				if (local_peaks.empty())
				{
					is_find_interval == true;
					sort(interval.begin(), interval.end());
					speed_block_interval = interval[interval.size() / 2];
					break;
				}
				//���������������������ü��ʱ�������µķ�ֵ�㣬�����÷�ֵ���ԭ������ɾ��
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

		vector<Mat> blocks; // �洢ÿ������
		//�ҵ�������ͼ����зָ�
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
		//�����Ⱦ�ֵ���Ϊ��ɫ�飬�����ƶ�����ɫ����ɫ��
		vector<int>colors_seq = basedYellowBlock(blocks);
		vector<bool>defect; //��¼�Ƿ�ȱʧ
		//�ж��Ƿ�ȱ��
		//����ÿһ���ݶ�
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
			//��ÿ��ͼ�ݶȽ��л���
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

			//���ؽ������꣬�����ʾͻ�俪ʼλ��
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
		//������
		for (int i = 0; i != colors_seq.size(); ++i)
		{
			if (colors_seq[i] == 0)
			{
				// trueΪȱ��
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