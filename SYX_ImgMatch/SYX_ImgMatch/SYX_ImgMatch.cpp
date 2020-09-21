// SYX_ImgMatch.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include "pch.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <highgui.h>
#include <stdlib.h>
#include <cv.h>

#define MYMAX 99999999
#define MYMIN -99999999

using namespace cv;
using namespace std;

Point click;

//鼠标获取坐标信息
void on_mouse(int event, int x, int y, int flags, void* img)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, CV_AA);
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		IplImage *timg = cvCloneImage((IplImage *)img);
		CvPoint pt = cvPoint(x, y);
		char temp[16];
		sprintf_s(temp, "(%d,%d)", x, y);
		cvPutText(timg, temp, pt, &font, CV_RGB(250, 0, 0));
		cvCircle(timg, pt, 2, cvScalar(255, 0, 0, 0), CV_FILLED, CV_AA, 0);
		click.x = x;
		click.y = y;
		cvShowImage("src", timg);
		cvReleaseImage(&timg);
	}
}

//交互操作：在目标图片中通过鼠标左键选取目标点
Mat Getori(Mat* src, Mat sr, Point target, int m, int n) {
	cv::Mat ori(cv::Size(m, n), CV_8UC1); //创建单通道的运算矩阵
	int k = m / 2;
	int L = n / 2;

	if (target.y - k >= 0 && target.y + k < src->rows&&target.x - L >= 0 && target.x + L < src->cols) {
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				ori.at<uchar>(i, j) = src->at<uchar>(target.y - k + i, target.x - L + j);
			}
		}

		rectangle(sr, Rect(target.x - L, target.y - k, m, n), Scalar(0, 0, 255), 1, 1, 0);
		sr.at<Vec3b>(target.y, target.x)[0] = 0;
		sr.at<Vec3b>(target.y, target.x)[1] = 0;
		sr.at<Vec3b>(target.y, target.x)[2] = 255;
		imshow("目标区域与目标点", sr);
		waitKey();
		return ori;
	}
	else {
		cout << "所选目标区域超过图像范围" << endl;
		exit;
	}
}

//交互操作:从待匹配图片中通过鼠标左键截取出搜索窗口
Mat Getfnd(Mat*src, Mat sr, Point seek, int K, int L) {
	cv::Mat fnd(cv::Size(K, L), CV_8UC1);
	int m = K / 2;
	int n = L / 2;

	if (seek.y - m >= 0 && seek.y + m < src->rows&&seek.x - n >= 0 && seek.y + n < src->cols) {
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < L; j++) {
				fnd.at<uchar>(i, j) = src->at<uchar>(seek.y - m + i, seek.x - n + j);
			}
		}
		rectangle(sr, Rect(seek.x - n, seek.y - m, K, L), Scalar(255, 0, 0), 1, 1, 0);
		imshow("搜索区域示意图", sr);
		waitKey();
		return fnd;
	}
	else {
		cout << "超出图片范围" << endl;
		exit;
	}
}

//相关系数的计算
double Getrou(Mat ori, Mat fnd, int c, int r, int m, int n) {
	double a = 0, b1 = 0, b2 = 0, c1 = 0, d1 = 0;
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			a += ori.at<uchar>(i, j) * fnd.at<uchar>(i + r, j + c);
			b1 += ori.at<uchar>(i, j);
			b2 += fnd.at<uchar>(i + r, j + c);
			c1 += ori.at<uchar>(i, j) * ori.at<uchar>(i, j);
			d1 += fnd.at<uchar>(i + r, j + c) * fnd.at<uchar>(i + r, j + c);
		}
	}

	double result = 0;
	result = (a - b1 * b2 / (m*n)) / (sqrt((c1 - b1 * b1 / (m*n))*(d1 - b2 * b2 / (m*n))));
	return result;
}

//兴趣点计算，并得到最大值对应的像素点
Point Getrous(int m, int n, int K, int L, Mat ori, Mat fnd, Point seek) {
	double res = MYMIN;
	int r = -1, s = -1;
	for (int i = 0; i < K - m; i++) {
		for (int j = 0; j < L - n; j++) {
			double a = Getrou(ori, fnd, i, j, m, n);
			if (a > res) {
				res = a;
				r = i;
				s = j;
			}
		}
	}
	Point result;
	result.x = seek.x - L / 2 + s + n / 2;
	result.y = seek.y - K / 2 + r + m / 2;
	return result;
}

//将匹配到的目标点和目标窗口显示在图片上
void showPoint(Mat sr, Point res, int m, int n) {
	sr.at<Vec3b>(res.y, res.x)[0] = 0;
	sr.at<Vec3b>(res.y, res.x)[1] = 0;
	sr.at<Vec3b>(res.y, res.x)[2] = 255;

	rectangle(sr, Rect(res.x - n / 2, res.y - m / 2, m, n), Scalar(0, 0, 255), 1, 1, 0);

	imshow("匹配的点与目标区域", sr);
	waitKey();
}

int main()
{
	int m, n, k, L;
	m = n = 29;
	k = L = 101;

	cout << "请通过鼠标点击您的目标点，可多次选择，确认后敲击回车键即可" << endl;
	IplImage *img = cvLoadImage("1_right.png", 1);
	cvNamedWindow("src", 1);
	cvSetMouseCallback("src", on_mouse, img);
	cvShowImage("src", img);
	cvWaitKey(0);
	cvDestroyAllWindows();
	cvReleaseImage(&img);

	Point target = click;

	cout << "请在立体相对中单击目标点的大致所在位置，可多次选择，确认后敲击回车键即可" << endl;
	IplImage *img1 = cvLoadImage("1_left.png", 1);
	cvNamedWindow("src", 1);
	cvSetMouseCallback("src", on_mouse, img1);
	cvShowImage("src", img1);
	cvWaitKey(0);
	cvDestroyAllWindows();
	cvReleaseImage(&img1);

	Point seek = click;

	//读取目标图像和待匹配图像
	Mat ori = imread("1_right.png", 0);
	Mat ori3 = imread("1_right.png", 1);
	Mat res = imread("1_left.png", 0);
	Mat res3 = imread("1_left.png", 1);

	Mat vori = Getori(&ori, ori3, target, m, n);
	Mat vfnd = Getfnd(&res, res3, seek, k, L);

	Point result = Getrous(m, n, k, L, vori, vfnd, seek);
	showPoint(res3, result, m, n);
}

// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门提示: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
