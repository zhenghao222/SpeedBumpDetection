/*TODO
 * improve edge linking
 * remove blobs whose axis direction doesnt point towards vanishing pt
 * Parallelisation
 * lane prediction
*/


#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <math.h>
#include <time.h>

using namespace std;
using namespace cv;


class LaneDetect
{
public:
	Mat currFrame; //stores the upcoming frame
	Mat temp;      //stores intermediate results
	Mat temp2;     //stores the final lane segments

	int diff, diffL, diffR;
	int laneWidth;
	int diffThreshTop;
	int diffThreshLow;
	int ROIrows;
	int vertical_left;
	int vertical_right;
	int vertical_top;
	int smallLaneArea;
	int longLane;
	int  vanishingPt;
	float maxLaneWidth;

	int middle_cols = 960 / 2;

	//to store various blob properties
	Mat binary_image; //used for blob removal
	int minSize;
	int ratio;
	float  contour_area;
	float blob_angle_deg;
	float bounding_width;
	float bounding_length;
	Size2f sz;
	vector< vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RotatedRect rotated_rect;
	vector<RotatedRect> rotatedRects;

	vector< vector<Point> > binary_contours;
	vector<Vec4i> binary_hierarchy;
	Mat startFrame1;
	LaneDetect(Mat startFrame,Mat rgb_startFrame)
	{
		//currFrame = startFrame;                                    //if image has to be processed at original size
		rgb_startFrame.copyTo(startFrame1);

		currFrame = Mat(320, 480, CV_8UC1, 0.0);                        //initialised the image size to 320x480
		resize(startFrame1, startFrame1, currFrame.size());
		resize(startFrame, currFrame, currFrame.size());             // resize the input to required size

		temp = Mat(currFrame.rows, currFrame.cols, CV_8UC1, 0.0);//stores possible lane markings
		temp2 = Mat(currFrame.rows, currFrame.cols, CV_8UC1, 0.0);//stores finally selected lane marks

		vanishingPt = currFrame.rows / 2;                           //for simplicity right now
		ROIrows = currFrame.rows - vanishingPt;               //rows in region of interest
		minSize = 0.00015 * (currFrame.cols*currFrame.rows);  //min size of any region to be selected as lane
		maxLaneWidth = 0.025 * currFrame.cols;                     //approximate max lane width based on image size
		smallLaneArea = 7 * minSize;
		longLane = 0.3 * currFrame.rows;
		ratio = 4;

		//these mark the possible ROI for vertical lane segments and to filter vehicle glare
		vertical_left = 2 * currFrame.cols / 5;
		vertical_right = 3 * currFrame.cols / 5;
		vertical_top = 2 * currFrame.rows / 3;

		namedWindow("lane", 2);
		namedWindow("midstep", 2);
		namedWindow("currframe", 2);
		namedWindow("laneBlobs", 2);
		getLane();
	}

	void updateSensitivity()
	{
		int total = 0, average = 0;
		for (int i = vanishingPt; i < currFrame.rows; i++)
			for (int j = 0; j < currFrame.cols; j++)
				total += currFrame.at<uchar>(i, j);
		average = total / (ROIrows*currFrame.cols);
		cout << "average : " << average << endl;
	}

	void getLane()
	{
		//medianBlur(currFrame, currFrame,5 );
		// updateSensitivity();
		//ROI = bottom half
		for (int i = vanishingPt; i < currFrame.rows; i++)
			for (int j = 0; j < currFrame.cols; j++)
			{
				temp.at<uchar>(i, j) = 0;
				temp2.at<uchar>(i, j) = 0;
			}

		imshow("currframe", currFrame);
		blobRemoval();
	}

	void markLane()
	{
		for (int i = vanishingPt; i < currFrame.rows; i++)
		{
			//IF COLOUR IMAGE IS GIVEN then additional check can be done
			// lane markings RGB values will be nearly same to each other(i.e without any hue)

			//min lane width is taken to be 5
			laneWidth = 5 + maxLaneWidth * (i - vanishingPt) / ROIrows;
			for (int j = laneWidth; j < currFrame.cols - laneWidth; j++)
			{

				diffL = currFrame.at<uchar>(i, j) - currFrame.at<uchar>(i, j - laneWidth);
				diffR = currFrame.at<uchar>(i, j) - currFrame.at<uchar>(i, j + laneWidth);
				diff = diffL + diffR - abs(diffL - diffR);

				//1 right bit shifts to make it 0.5 times
				diffThreshLow = currFrame.at<uchar>(i, j) >> 1;
				//diffThreshTop = 1.2*currFrame.at<uchar>(i,j);

				//both left and right differences can be made to contribute
				//at least by certain threshold (which is >0 right now)
				//total minimum Diff should be atleast more than 5 to avoid noise
				if (diffL > 0 && diffR > 0 && diff > 5)
					if (diff >= diffThreshLow /*&& diff<= diffThreshTop*/)
						temp.at<uchar>(i, j) = 255;
			}
		}

	}

	void blobRemoval()
	{
		markLane();

		// find all contours in the binary image
		temp.copyTo(binary_image);
		//Mat kernel = getStructuringElement(MORPH_RECT, Size(2 * 1 + 1, 2 * 1 + 1), Point(1, 1));
		//dilate(binary_image, binary_image, kernel);
		findContours(binary_image, contours,
			hierarchy, RETR_CCOMP,
			CHAIN_APPROX_SIMPLE);

		Mat counters_show;
		startFrame1.copyTo(counters_show);
		for (size_t i = 0; i < contours.size(); ++i)
		{
			drawContours(counters_show, contours, i, Scalar(0,0,255), 1, 8);
		}
		namedWindow("curr_counters", WINDOW_NORMAL);
		imshow("curr_counters", counters_show);
		

		// for removing invalid blobs
		if (!contours.empty())
		{
			for (size_t i = 0; i < contours.size(); ++i)
			{
				//====conditions for removing contours====//

				contour_area = contourArea(contours[i]);

				//blob size should not be less than lower threshold
				if (contour_area > minSize)
				{
					rotated_rect = minAreaRect(contours[i]);
					sz = rotated_rect.size;
					bounding_width = sz.width;
					bounding_length = sz.height;


					//openCV selects length and width based on their orientation
					//so angle needs to be adjusted accordingly
					blob_angle_deg = rotated_rect.angle;
					if (bounding_width < bounding_length)
						blob_angle_deg = 90 + blob_angle_deg;

					//if such big line has been detected then it has to be a (curved or a normal)lane
					if (bounding_length > longLane || bounding_width > longLane)
					{
						drawContours(currFrame, contours, i, Scalar(255), FILLED, 8);
						drawContours(temp2, contours, i, Scalar(255), FILLED, 8);
					}

					//angle of orientation of blob should not be near horizontal or vertical
					//vertical blobs are allowed only near center-bottom region, where centre lane mark is present
					//length:width >= ratio for valid line segments
					//if area is very small then ratio limits are compensated
					else if ((blob_angle_deg <-10 || blob_angle_deg >-10) &&
						((blob_angle_deg > -70 && blob_angle_deg < 70) ||
						(rotated_rect.center.y > vertical_top &&
							rotated_rect.center.x > vertical_left && rotated_rect.center.x < vertical_right)))
					{

						if ((bounding_length / bounding_width) >= ratio || (bounding_width / bounding_length) >= ratio
							|| (contour_area < smallLaneArea && ((contour_area / (bounding_width*bounding_length)) > .75) &&
							((bounding_length / bounding_width) >= 2 || (bounding_width / bounding_length) >= 2)))
						{
							drawContours(currFrame, contours, i, Scalar(255), FILLED, 8);
							drawContours(temp2, contours, i, Scalar(255), FILLED, 8);
						}

					}

					
				}
			}
		}
		imshow("midstep", temp);
		imshow("laneBlobs", temp2);
		imshow("lane", currFrame);
		//cout << temp2 << endl;
		resize(temp2, temp2, Size(1279, 960));
		Mat temp2_rgbtemp2(temp2.size(), CV_8UC3);
		vector< vector<Point> > temp2_contours;
		vector<Vec4i> temp2_hierarchy;
		findContours(temp2, temp2_contours, temp2_hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
		if (!temp2_contours.empty())
		{
			for (size_t i = 0; i < temp2_contours.size(); ++i)
			{
				rotated_rect = minAreaRect(temp2_contours[i]);
				Point2f vertex[4];
				rotated_rect.points(vertex);
				for (int i = 0; i < 4; i++)
				{
					line(temp2_rgbtemp2, vertex[i], vertex[(i + 1) % 4], Scalar(0, 0, 255), 1, 8);
				}
				rotatedRects.push_back(rotated_rect);
			}
		}
		imshow("rgbn_temp2", temp2_rgbtemp2);
		

	}


	void nextFrame(Mat &nxt)
	{
		//currFrame = nxt;                        //if processing is to be done at original size

		resize(nxt, currFrame, currFrame.size()); //resizing the input image for faster processing
		getLane();
	}

	Mat getResult()
	{
		return temp2;
	}


	vector<RotatedRect> edgeExtrction()
	{
		RotatedRect leftLane, rightLane;
		vector<RotatedRect> left_rightLane;
		sort(rotatedRects.begin(), rotatedRects.end(), [=](const RotatedRect& rect1, const RotatedRect& rect2) {return rect1.size.height * rect1.size.width > rect2.size.height * rect2.size.width; });
		if (!rotatedRects.empty())
		{
			if (rotatedRects[0].center.x < middle_cols)
			{
				leftLane = rotatedRects[0];
				rightLane = rotatedRects[1];
			}
			else
			{
				leftLane = rotatedRects[1];
				rightLane = rotatedRects[0];
			}
			left_rightLane.push_back(leftLane);
			left_rightLane.push_back(rightLane);
			cout << left_rightLane[0].center << endl;
			cout << left_rightLane[1].center << endl;
		}
		return left_rightLane;
	}
};//end of class LaneDetect



