// FirstApp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>

#pragma comment(lib,"opencv_core3414d.lib")
#pragma comment(lib,"opencv_highgui3414d.lib")
#pragma comment(lib,"opencv_imgcodecs3414d.lib")
#pragma comment(lib,"opencv_videoio3414d.lib")
#pragma comment(lib,"opencv_imgproc3414d.lib")



//看图片

//int main()
//{
//    cv::Mat img = cv::imread(R"(C:\Users\zds\Desktop\laojun_pic.png)",-1);
//    if (img.empty()) {
//        return -1;
//    }
//
//    cv::namedWindow("Example1", cv::WINDOW_AUTOSIZE);
//    cv::imshow("Example1", img);
//    cv::waitKey(0);
//    cv::destroyWindow("Example1");
//    return 0;
//}



//播放视频
//
//int main()
//{
//    cv::namedWindow("Example3", cv::WINDOW_AUTOSIZE);
//    cv::VideoCapture cap;
//    cap.open(R"(F:\迅雷下载\[韩国三级][我朋友的老姐].2016.Uncut.720p.HDRip.H264-ob.mp4)");
//    cv::Mat frame;
//    
//    for (;;) {
//        cap >> frame;
//        if (frame.empty()) {
//            break;
//        }
//        cv::imshow("Example3", frame);
//        if (cv::waitKey(33) >= 0) {
//            break;
//        }
//    }
//    return 0;
//}

//播放视频带滑块
//
//int g_slider_position = 0;
//int g_run = 1, g_dontset = 0;
//cv::VideoCapture g_cap;
//
//void onTrackbarSlider(int pos, void*) {
//    g_cap.set(cv::CAP_PROP_POS_FRAMES, pos);
//    if (!g_dontset) {
//        g_run = 1;
//        g_dontset = 0;
//    }
//}
//
//int main()
//{
//    cv::namedWindow("Example2_4", cv::WINDOW_AUTOSIZE);
//    g_cap.open(R"(F:\迅雷下载\[韩国三级][我朋友的老姐].2016.Uncut.720p.HDRip.H264-ob.mp4)");
//    int frames = g_cap.get(cv::CAP_PROP_FRAME_COUNT);
//    int tmpw = g_cap.get(cv::CAP_PROP_FRAME_WIDTH);
//    int tmph = g_cap.get(cv::CAP_PROP_FRAME_HEIGHT);
//    std::cout << "Video has " << frames << " frames of dimensions(" << tmpw << " ," << tmph << ")." << std::endl;
//    cv::createTrackbar("Position", "Example2_4", &g_slider_position, frames, onTrackbarSlider);
//
//    cv::Mat frame;
//    for (;;) {
//        if (g_run != 0) {
//            g_cap >> frame;
//            if (frame.empty()) {
//                break;
//            }
//            int current_pos = (int)g_cap.get(cv::CAP_PROP_POS_FRAMES);
//            g_dontset = 1;
//            cv::setTrackbarPos("Position", "Example2_4", current_pos);
//            cv::imshow("Example2_4", frame);
//            g_run -= 1;
//        }
//
//        char c = cv::waitKey(10);
//        if (c == 's') {
//            g_run = 1;
//            std::cout << "Single step,run = " << g_run << std::endl;
//        }
//
//        if (c == 'r') {
//            g_run = -1;
//            std::cout << "Run mode,run = " << g_run << std::endl;
//        }
//
//        if (c == 27) {
//            break;
//        }
//    }
//
//    return 0;
//}


//cv::threshold用法

//void sum_rgb(const cv::Mat& src, cv::Mat & dst) {
//    //分割图像到颜色盘
//    std::vector<cv::Mat> planes;
//    cv::split(src, planes);
//
//    cv::Mat b = planes[0], g = planes[1], r = planes[2], s;
//
//    //添加同等加权的 rgb 值
//    cv::addWeighted(r, 1. / 3., g, 11. / 3., 0.0, s);
//    cv::addWeighted(s,1.,b,1./3.,0.0,s);
//
//    //删除大于100的
//    cv::threshold(s, dst, 100, 100, cv::THRESH_TRUNC);
//
//}
//void sum_rgb(const cv::Mat& src, cv::Mat& dst) {
//    //分割图像到颜色盘
//    std::vector<cv::Mat> planes;
//    cv::split(src, planes);
//
//    cv::Mat b = planes[0], g = planes[1], r = planes[2];
//    //累积 分离的planes，组合然后threshold
//    cv::Mat s = cv::Mat::zeros(b.size(), CV_32F);
//    cv::accumulate(b, s);
//    cv::accumulate(g, s);
//    cv::accumulate(r, s);
//
//    //truncate value above 100 
//    cv::threshold(s, s, 100, 100, cv::THRESH_TRUNC);
//    s.convertTo(dst, b.type());
//
//}
//
//int main()
//{
//    cv::Mat src = cv::imread(R"(C:\Users\zds\Desktop\laojun_pic.png)", -1),dst;
//    sum_rgb(src, dst);
//    cv::imshow("text", dst);
//    cv::waitKey(0);
//    return 0;
//}
//


//阈值与自适应阈值
//
//
//int main()
//{
//    double fixed_threshold = 100;
//    int threshold_type = cv::THRESH_BINARY;
//    int ataptive_method = cv::ADAPTIVE_THRESH_MEAN_C;
//
//    int block_size = 15;
//    double offset = 10;
//
//    cv::Mat src = cv::imread(R"(C:\Users\zds\Desktop\laojun_pic.png)",cv::IMREAD_GRAYSCALE),dst;
//    cv::Mat it, iat;
//    cv::threshold(src, it, fixed_threshold, 255, threshold_type);
//    cv::adaptiveThreshold(src, iat, 255, ataptive_method, threshold_type, block_size, offset);
//
//
//
//    cv::imshow("text", src);
//    cv::imshow("threshold", it);
//    cv::imshow("adaptiveThreshold", iat);
//    cv::waitKey(0);
//    return 0;
//}



//视频教程图像的淹膜操作
//
//
//int main()
//{
//    cv::Mat src = cv::imread(R"(C:\Users\zds\Desktop\laojun_pic.png)");
//    cv::imshow("text", src);
//    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
//
//    //int cols = (src.cols - 1) * src.channels();//宽
//    //int offsetx = src.channels();
//    //int rows = src.rows;
//    //for (int row = 1; row < src.rows - 1; row++) {
//    //    const uchar * current_row_ptr = src.ptr<uchar>(row);
//    //    const uchar * previous_row_ptr = src.ptr<uchar>(row - 1);
//    //    const uchar * next_row_ptr = src.ptr<uchar>(row + 1);
//
//    //    uchar* output = dst.ptr<uchar>(row);
//    //    for (int col = offsetx; col < (cols - offsetx); col++) {
//    //        output[col] = cv::saturate_cast<uchar>(5 * current_row_ptr[col] - (current_row_ptr[col - offsetx] + current_row_ptr[col] + offsetx + previous_row_ptr[col] + next_row_ptr[col]));
//    //    }
//    //}
//
//    //使用filter2D
//    cv::Mat kernal = (cv::Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//    cv::filter2D(src, dst, src.depth(), kernal);
//
//    cv::namedWindow("out win", CV_WINDOW_AUTOSIZE);
//    imshow("out win", dst);
//    cv::waitKey(0);
//    return 0;
//}

//
//int main()
//{
//	cv::Mat src = cv::imread(R"(C:\Users\zds\Desktop\laojun_pic.png)");
//	cv::Mat dst;
//	dst = cv::Mat(src.size(), src.type());
//	dst = cv::Scalar(127, 0, 255);//设置纯粉
//	cv::imwrite(R"(C:\Users\zds\Desktop\dst.png)", dst );
//
//	uchar * val = dst.ptr<uchar>(0, 0);//127,0,255
//	uchar* val2 = dst.ptr<uchar>(0, 1); //127.0.255
//
//	std::cout << dst << std::endl;
//
//
//	cv::imshow("dst", dst);
//	cv::waitKey(0);
//	return 0;
//}

//视频教程：图像的混合
//int main()
//{
//	cv::Mat src1 = cv::imread("1.jpg");
//	cv::Mat src2 = cv::imread("2.jpg");
//
//	cv::imshow("src1", src1);
//	cv::imshow("src2", src2);
//
//	cv::Mat src3;
//	if (src1.rows == src2.rows && src1.cols == src2.cols && src1.type() == src2.type()) {
//		cv::addWeighted(src1, 0.5, src2, 1 - 0.5, 0.0, src3);
//		//cv::add(src1, src2,src3);
//		//cv::multiply(src1, src2, src3);
//		//v::imshow("src3", src3);
//	}
//
//	cv::waitKey(0);
//	return 0;
//}

//视频教程：调整亮度和对比度
//int main()
//{
//	cv::Mat src1 = cv::imread("1.jpg");
//	int rows = src1.rows;
//	int width = src1.cols;
//
//	float alpha = 1.2f;
//	float beta = 30;
//
//	cv::Mat src2 = cv::Mat::zeros(src1.size(), src1.type());
//	for (int row = 0; row < rows; row++) {
//		for (int col = 0; col < width; col++) {
//			if (src1.channels() == 3) {
//				const auto &  rgb = src1.at<cv::Vec3b>(row, col);
//				auto& dst_rgb = src2.at<cv::Vec3b>(row, col);
//				
//				dst_rgb[0] = cv::saturate_cast<uchar>(rgb[0] * alpha + beta);
//				dst_rgb[1] = cv::saturate_cast<uchar>(rgb[1] * alpha + beta);
//				dst_rgb[2] = cv::saturate_cast<uchar>(rgb[2] * alpha + beta);
//			}
//			else if (src1.channels() == 1) {
//				auto v = src1.at<uchar>(row, col);
//				auto& dst_rgb = src2.at<uchar>(row, col);
//				dst_rgb = v * alpha + beta;
//			}
//		}
//	}
//
//	cv::imshow("image1", src1);
//	cv::imshow("image2", src2);
//	
//	cv::waitKey(0);
//	return 0;
//}

//绘制开关与文字
//void MyLine(cv::Mat &  mat)
//{
//	cv::Point p1 = { 20,30 };
//	cv::Point p2;
//	p2.x = 300;
//	p2.y = 300;
//	cv::Scalar color = { 0,0,255 };
//	cv::line(mat, p1, p2, color, 1, cv::LINE_8);
//}
//void MyRect(cv::Mat & mat)
//{
//	cv::Rect rc = { 200,100,300,300 };
//	cv::Scalar color = { 255,0,0 };
//	cv::rectangle(mat, rc, color);
//}
//
//void PutText(cv::Mat& mat)
//{
//	cv::putText(mat, "Hello world", cv::Point(300, 300), CV_FONT_HERSHEY_COMPLEX, 1.0, cv::Scalar(255, 0, 0));
//}
//int main()
//{
//	cv::Mat src1 = cv::imread("1.jpg");
//	MyLine(src1);
//	MyRect(src1);
//	PutText(src1);
//	cv::imshow("image1", src1);
//	cv::waitKey(0);
//	return 0;
//}


////模糊图像
////模糊图像二
//int main()
//{
//	cv::Mat src1 = cv::imread("Me.png");
//	cv::Mat dst1,dst2;
//
//	//cv::blur(src1, dst1, cv::Size(11, 11));
//	//cv::GaussianBlur(src1, dst2, cv::Size(11,11), 11, 11);
//
//	cv::namedWindow("input image", CV_WINDOW_AUTOSIZE);
//	cv::imshow("input image", src1);
//
//	//cv::medianBlur(src1, dst1, 3);
//	cv::bilateralFilter(src1, dst1, 15, 100, 3);
//	cv::Mat kernel = (cv::Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1,0);
//	cv::filter2D(dst1, dst2, -1, kernel, cv::Point(-1, -1), 0);
//	cv::imshow("image2", dst1);
//	cv::imshow("image3", dst2);
//	cv::waitKey(0);
//	return 0;
//}

//
//
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")
////膨胀或塌陷
////
//int element_size = 3;
//int max_size = 21;
//
//cv::Mat src1;
//cv::Mat dst1;
//void CallBackDemo(int, void*) {
//	int s = element_size * 2 + 1;
//	cv::Mat structure_elem = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(s, s));
//	cv::dilate(src1, dst1, structure_elem);
//	//cv::erode(src1, dst1, structure_elem);
//	cv::imshow("output image", dst1);
//}
//
//
//int main()
//{
//	src1 = cv::imread("gray.png");
//	
//	cv::namedWindow("input image", CV_WINDOW_AUTOSIZE);
//	cv::imshow("input image", src1);
//
//	cv::namedWindow("output image", CV_WINDOW_AUTOSIZE);
//
//
//	cv::createTrackbar("Element Size:","output image", &element_size, max_size, CallBackDemo);
//	CallBackDemo(0, 0);
//	cv::waitKey(0);
//	return 0;
//}





//int main()
//{
//	//开操作：取消小的对象
//	//过程：先腐蚀后膨胀
//
//	//闭操作：功能填充小洞
//	//过程：先膨胀后腐蚀
//	//cv::Mat src1 = cv::imread("gray.png");
//	//std::string input_image_title = "Input Image", out_image_title = "Output Image";
//	//cv::namedWindow(input_image_title, CV_WINDOW_AUTOSIZE);
//	//cv::imshow(input_image_title,src1);
//
//	//cv::Mat dst;
//	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(50,50), cv::Point(-1, -1));
//	//cv::morphologyEx(src1, dst, CV_MOP_CLOSE, kernel);
//	//cv::morphologyEx(src1, dst, CV_MOP_OPEN, kernel);
//
//
//	//cv::namedWindow(out_image_title, CV_WINDOW_AUTOSIZE);
//	//cv::imshow(out_image_title, dst);
//
//
//	//梯度：膨胀减去腐
//	//顶帽：源图像与开操作之间的差值置图像 结果：去掉的大的区域
//	cv::Mat src1 = cv::imread("gray.png");
//	std::string input_image_title = "Input Image", out_image_title = "Output Image";
//	cv::namedWindow(input_image_title, CV_WINDOW_AUTOSIZE);
//	cv::imshow(input_image_title,src1);
//
//	cv::Mat dst;
//	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(11,11), cv::Point(-1, -1));
//	//cv::morphologyEx(src1, dst, CV_MOP_GRADIENT, kernel); //梯度
//	//cv::morphologyEx(src1, dst, CV_MOP_TOPHAT, kernel); //顶帽
//	cv::morphologyEx(src1, dst, CV_MOP_BLACKHAT, kernel); //黑帽
//
//
//	cv::namedWindow(out_image_title, CV_WINDOW_AUTOSIZE);
//	cv::imshow(out_image_title, dst);
//
//
//
//	cv::waitKey(0);
//	return 0;
//}




//提取水平和垂直线

//int main()
//{
//	cv::Mat src1 = cv::imread("Bin1.png");
//	cv::namedWindow("INPUT_WINDOW", CV_WINDOW_AUTOSIZE);
//	cv::imshow("INPUT_WINDOW", src1);
//
//	//变成一张灰度图像
//	cv::Mat gray;
//	cv::cvtColor(src1, gray, CV_BGR2GRAY);
//	cv::imshow("GRAY_WINDOW", gray);
//
//	//变成一张二值图像
//	cv::Mat bin_img;
//	cv::adaptiveThreshold(~gray, bin_img, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);
//	cv::imshow("BIN_WINDOW", bin_img);
//
//
//	//开操作
//	cv::Mat hline = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(src1.cols / 16, 1), cv::Point(-1, -1));
//	cv::Mat vline = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,src1.rows / 16), cv::Point(-1, -1));
//
//	//以下操作的结果是 横线保留了，竖线擦除了
//	//cv::Mat dst, temp;
//	//cv::erode(bin_img, temp, hline);
//	//cv::dilate(temp, dst, hline);
//	//cv::bitwise_not(dst, dst); //背景变成白色
//	//cv::imshow("RESULT", dst);
//
//	//以下操作的结果是 竖线保留了，横线擦除了
//	cv::Mat dst, temp;
//	cv::erode(bin_img, temp, vline);
//	cv::dilate(temp, dst, vline);
//	cv::bitwise_not(dst, dst); //背景变成白色
//	cv::imshow("RESULT", dst);
//
//
//
//
//	cv::waitKey(0);
//	return 0;
//}

//去除图像
//int main()
//{
//	cv::Mat src1 = cv::imread("Test.png");
//	cv::namedWindow("INPUT_WINDOW", CV_WINDOW_AUTOSIZE);
//	cv::imshow("INPUT_WINDOW", src1);
//
//	//变成一张灰度图像
//	cv::Mat gray;
//	cv::cvtColor(src1, gray, CV_BGR2GRAY);
//	cv::imshow("GRAY_WINDOW", gray);
//
//	//变成一张二值图像
//	cv::Mat bin_img;
//	cv::adaptiveThreshold(~gray, bin_img, 255, CV_ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 15, -2);
//	cv::imshow("BIN_WINDOW", bin_img);
//
//
//	//开操作
//	//取消验证码干扰线
//	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
//
//	//以下操作的结果是 横线保留了，竖线擦除了
//	cv::Mat dst, temp;
//	cv::erode(bin_img, temp, kernel);
//	cv::dilate(temp, dst, kernel);
//	cv::bitwise_not(dst, dst); //背景变成白色
//	cv::imshow("RESULT", dst);
//
//
//
//
//	cv::waitKey(0);
//	return 0;
//}


//基本阀值操作

//
//
//cv::Mat src, dst,gray;
//
//int threshold_value = 127;
//int threshold_max = 255;
//
//int type_value = 2;
//int type_max = 4;
//
//
//void Threshold_Demo(int, void*);
//
//int main()
//{
//    src = cv::imread("test.jpg");
//    cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
//    cv::namedWindow("Output Image", cv::WINDOW_AUTOSIZE);
//
//
//    cv::imshow("Input Image", src);
//    cv::createTrackbar("Threshold Value:", "Output Image", &threshold_value, threshold_max, Threshold_Demo);
//    cv::createTrackbar("Type Value     :", "Output Image", &type_value, type_max, Threshold_Demo);
//
//
//    cv::waitKey(0);
//    std::cout << "Hello World!\n";
//    return 0;
//}
//
//void Threshold_Demo(int, void*)
//{
//    cv::cvtColor(src, gray, CV_BGR2GRAY);
//    cv::threshold(gray, dst, 0, 255,CV_THRESH_OTSU |   type_value);
//    cv::imshow("Output Image", dst);
//}


//图像金子塔 上采样与降采样


//int main()
//{
//	cv::Mat src = cv::imread("test.jpg");
//	cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
//	cv::imshow("Input Image", src);
//
//	cv::Mat dst, dst2;
//
//	cv::pyrUp(src, dst, cv::Size(src.cols * 2, src.rows * 2)); // 上采样
//	cv::imshow("Output Image", dst);
//
//	cv::pyrDown(src, dst2, cv::Size(src.cols / 2, src.rows / 2)); // 降采样
//	cv::imshow("Output Image2", dst2);
//
//	//DOG
//	cv::Mat g1, g2, gray_src, dog_img;
//	cv::cvtColor(src, gray_src, CV_BGR2GRAY);
//	cv::GaussianBlur(gray_src, g1, cv::Size(5, 5), 0, 0);
//	cv::GaussianBlur(g1, g2, cv::Size(5, 5), 0, 0);
//	cv::subtract(g1, g2, dog_img);
//
//	//归一化显示 如果不弄，会显示很暗
//	cv::normalize(dog_img, dog_img, 255, 0, cv::NORM_MINMAX);
//	cv::imshow("Dog Img", dog_img);
//	cv::waitKey(0);
//	std::cout << "Hello World!\n";
//	return 0;
//}



//自定义线性滤波


//int main()
//{
//	cv::Mat src = cv::imread("Me.png");
//	
//	cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
//	cv::imshow("Input Image", src);
//
//	//robot 算子
//	//x方向
//	{
//		cv::Mat dst1;
//		cv::Mat kernel_x = (cv::Mat_<int>(2, 2) << 1, 0, 0, -1);
//		cv::filter2D(src, dst1, -1, kernel_x, cvPoint(-1, -1), 0.0);
//		cv::imshow("Output Image1", dst1);
//	}
//	{
//		//y方向
//		cv::Mat dst2;
//		cv::Mat kernel_y = (cv::Mat_<int>(2, 2) << 0, 1, -1, 0);
//		cv::filter2D(src, dst2, -1, kernel_y, cvPoint(-1, -1), 0.0);
//		cv::imshow("Output Image2", dst2);
//	}
//
//	//sobel算了子
//		//x方向
//	{
//		cv::Mat dst1;
//		cv::Mat kernel_x = (cv::Mat_<int>(3, 3) << -1,0,1,-2,0,2,-1,0,1);
//		cv::filter2D(src, dst1, -1, kernel_x, cvPoint(-1, -1), 0.0);
//		cv::imshow("Output Image3", dst1);
//	}
//	{
//		//y方向
//		cv::Mat dst2;
//		cv::Mat kernel_y = (cv::Mat_<int>(3, 3) << -1,-2,-1,0,0,0,1,2,1);
//		cv::filter2D(src, dst2, -1, kernel_y, cvPoint(-1, -1), 0.0);
//		cv::imshow("Output Image4", dst2);
//	}
//
//	//拉普拉斯算子
//	{
//		cv::Mat dst1;
//		cv::Mat kernel_x = (cv::Mat_<int>(3, 3) << 0,-1,0,-1,4,-1,0,-1,0);
//		cv::filter2D(src, dst1, -1, kernel_x, cvPoint(-1, -1), 0.0);
//		cv::imshow("Output Image5", dst1);
//	}
//
//	//自定义核
//
//	{
//		int index = 0;
//		int ksize = 3;
//		cv::Mat dst;
//		while (true) {
//			ksize = 4 + (index % 5) * 2 + 1;
//			cv::Mat kernel = cv::Mat::ones(cv::Size(ksize, ksize), CV_32F) / (float)(ksize * ksize);
//			cv::filter2D(src, dst, -1, kernel);
//			index++;
//			cv::imshow("Output Image6", dst);
//			cv::waitKey(500);
//		}
//	}
//
//	cv::waitKey(0);
//	return 0;
//}

//图像边缘处理

//int main()
//{
//	cv::Mat src = cv::imread("Me.png");
//	cv::Mat dst;
//
//	cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
//	cv::imshow("Input Image", src);
//
//	/*
//	int top = src.rows / 20;
//	int bottom = src.rows / 20;
//	int left = src.cols / 20;
//	int right = src.cols / 20;
//
//	cv::RNG rng(132);
//
//	int borad_type = cv::BorderTypes::BORDER_CONSTANT;
//	while (true) {
//		auto key = cv::waitKey(500);
//		if (key == 'r') {
//			borad_type += 1;
//			if (borad_type >= cv::BorderTypes::BORDER_TRANSPARENT) {
//				borad_type = cv::BorderTypes::BORDER_CONSTANT;
//			}
//		}
//		cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255),rng.uniform(0, 255));
//		cv::copyMakeBorder(src, dst, top, bottom, left, right, borad_type, color);
//		cv::imshow("RESULT",dst);
//	}
//	*/
//
//	cv::GaussianBlur(src, dst, cv::Size(5, 5), 0, 0, cv::BorderTypes::BORDER_WRAP);
//	cv::imshow("Result", dst);
//	cv::waitKey(0);
//	return 0;
//}




//sobel 算子

//int main()
//{
//	cv::Mat src = cv::imread("Me.png");
//
//	cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
//	cv::imshow("Input Image", src);
//
//	cv::Mat dst;
//	cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);
//
//	cv::Mat gray_src;
//	cv::cvtColor(dst, gray_src, CV_BGR2GRAY);
//
//
//	cv::Mat xgrad, ygrad;
//	//cv::Sobel(gray_src, xgrad, CV_16S, 1, 0, 3);
//	//cv::Sobel(gray_src, ygrad, CV_16S, 0, 1, 3);
//	cv::Scharr(gray_src, xgrad, CV_16S, 1, 0);
//	cv::Scharr(gray_src, ygrad, CV_16S, 0, 1);
//	cv::convertScaleAbs(xgrad, xgrad);
//	cv::convertScaleAbs(ygrad, ygrad);
//	cv::imshow("xgrad", xgrad);
//	cv::imshow("ygrad", ygrad);
//
//	cv::imshow("Output", gray_src);
//
//	//cv::Mat xygrad;
//	//cv::addWeighted(xgrad, 0.5, ygrad, 0.5, 0, xygrad);
//	//cv::imshow("Fin",xygrad);
//
//	//手工处理代理addweighted竟然比addweigted好
//	cv::Mat xygrad = cv::Mat(xgrad.size(), xgrad.type());
//	int width = xygrad.cols;
//	int height = xygrad.rows;
//	for (int row = 0; row < height; row++)
//	{
//		for (int col = 0; col < width; col++){
//			int xg = xgrad.at<uchar>(row, col);
//			int yg = xgrad.at<uchar>(row, col);
//			int xy = xg + yg;
//
//			xgrad.at<uchar>(row, col) = cv::saturate_cast<uchar>( xy);
//		}
//	}
//	cv::addWeighted(xgrad, 0.5, ygrad, 0.5, 0, xygrad);
//	cv::imshow("Fin",xygrad);
//
//	cv::waitKey();
//	return 0;
//}



//Laplase算子

//int main()
//{
//	cv::Mat src = cv::imread("Me.png");
//
//	cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
//	cv::imshow("Input Image", src);
//
//	cv::Mat dst;
//	cv::GaussianBlur(src, dst, cv::Size(3, 3), 0, 0);
//
//	cv::Mat gray_src;
//	cv::cvtColor(dst, gray_src, CV_BGR2GRAY);
//
//
//	cv::Mat edge_image;
//	cv::Laplacian(gray_src, edge_image, CV_16S, 3);
//	cv::convertScaleAbs(edge_image, edge_image);
//
//	cv::imshow("Result", edge_image);
//
//
//	cv::threshold(edge_image, edge_image, 0, 255, CV_THRESH_OTSU | CV_THRESH_BINARY);
//
//	cv::imshow("Result", edge_image);
//
//
//	cv::waitKey();
//	return 0;
//}



//Canny边缘检测算法
//
//
//int t1_value = 50;
//int max_value = 255;
//void Canny_demo(int, void*);
//
//cv::Mat src;
//cv::Mat gray_src;
//
//int main()
//{
//	src = cv::imread("Me.png");
//	cv::imshow("Src", src);
//
//	cv::cvtColor(src, gray_src, CV_BGR2GRAY);
//
//	cv::namedWindow("Result Window", cv::WINDOW_AUTOSIZE);
//
//	cv::createTrackbar("Value", "Result Window", &t1_value, max_value, Canny_demo);
//	Canny_demo(0, 0);
//
//
//
//
//	cv::waitKey();
//	return 0;
//}
//
//void Canny_demo(int, void*)
//{
//	cv::Mat blur_mat;
//	cv::blur(gray_src, blur_mat, cv::Size(3, 3));
//
//	cv::Mat edge_output;
//	cv::Canny(blur_mat, edge_output, t1_value, t1_value * 2, 3);
//
//	//cv::Mat dst;
//	//dst.create(src.size(), src.type());
//	//src.copyTo(dst, edge_output);
//	cv::imshow("Result Window", edge_output);
//}

//霍夫直线变换

//int main()
//{
//	cv::Mat src = cv::imread("Bin1.png");
//
//	cv::imshow("Input Image", src);
//
//	cv::Mat src_gray;
//	cv::Canny(src, src_gray, 100, 200);
//	cv::Mat dst;
//	cv::cvtColor(src_gray, dst, CV_GRAY2BGR);
//	cv::imshow("Result", src_gray);
//
//	std::vector<cv::Vec4f> plines;
//	cv::HoughLinesP(src_gray, plines, 1, CV_PI / 180.0, 10, 0, 10);
//	cv::Scalar color = cv::Scalar(0, 0, 255);
//	for (const auto & line : plines) {
//		cv::line(dst, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]),color,3,cv::LINE_AA);
//	}
//
//	cv::imshow("Result2", dst);
//
//	cv::waitKey(0);
//	return 0;
//}


//霍夫圆变换
//int main()
//{
//	cv::Mat src = cv::imread("cell.JFIF");
//	cv::imshow("Input Image", src);
//
//	cv::Mat m1;
//	cv::medianBlur(src, m1, 5);
//
//	cv::imshow("result1", m1);
//
//	cv::Mat m2;
//	cv::cvtColor(m1, m2, CV_BGR2GRAY);
//
//	cv::Mat m3;
//	std::vector<cv::Vec3f> circles;
//	cv::HoughCircles(m2, circles, CV_HOUGH_GRADIENT, 1, 10, 100, 30, 5, 50);
//
//
//	cv::Mat dst;
//	src.copyTo(dst);
//	for (auto& it : circles) {
//		cv::circle(dst, cv::Point(it[0], it[1]), it[2], cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
//		cv::circle(dst, cv::Point(it[0], it[1]), 2, cv::Scalar(98, 23, 255), 2, cv::LINE_AA);
//	}
//
//	cv::imshow("result1", dst);
//	cv::waitKey(0);
//	return 0;
//}


//像素重映射
//void update_map(void);
//int index = 0;
//cv::Mat map_x, map_y;
//cv::Mat src;
//int main()
//{
//	src = cv::imread("cell.JFIF");
//	cv::imshow("Input Image", src);
//
//	
//	map_x.create(src.size(), CV_32FC1);
//	map_y.create(src.size(), CV_32FC1);
//	index = 1;
//	update_map();
//	cv::Mat dst;
//	cv::remap(src, dst, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT,cv::Scalar(0,255,255));
//	cv::imshow("Output Image", dst);
//
//	cv::waitKey(0);
//	return 0;
//}
//
//
//void update_map(void)
//{
//	for (int i = 0; i < src.rows; i++) {
//		for (int j = 0; j < src.cols; j++) {
//			switch (index)
//			{
//			case 0:
//				if (j > (src.cols * 0.25) && j <= (src.cols * 0.75) && i >(src.rows * 0.25) && i <= (src.rows *0.75)) {
//					map_x.at<float>(i,j) = 2 * (j - (src.cols * 0.25) + 0.5);
//					map_y.at<float>(i, j) = 2 * (i - (src.rows * 0.25) + 0.5);
//				}
//				else {
//					map_x.at<float>(i,j) = 0;
//					map_y.at<float>(i, j) = 0;
//				}
//				break;
//			case 1:
//				map_x.at<float>(i, j) = src.cols - j - 1;
//				map_y.at<float>(i, j) = i;
//				break;
//			case 2:
//				map_x.at<float>(i, j) = j;
//				map_y.at<float>(i, j) = src.rows - i - 1;
//				break;
//			case 3:
//				map_x.at<float>(i, j) = src.cols - j - 1;
//				map_y.at<float>(i, j) = src.rows - i - 1;
//				break;
//			default:
//				break;
//			}
//		}
//	}
//}


//直方图均衡化
//int main()
//{
//	cv::Mat	src = cv::imread("Me.png");
//	cv::Mat gray;
//	cv::cvtColor(src, gray, CV_BGR2GRAY);
//	cv::imshow("Gray", gray);
//	cv::Mat dst;
//	cv::equalizeHist(gray, dst);
//	cv::imshow("Result",dst);
//	cv::waitKey();
//	return 0;
//}



//int main()
//{
//	cv::Mat	src = cv::imread("Me.png");
//
//	std::vector<cv::Mat> bgr_planes;
//	cv::split(src, bgr_planes);
//	cv::imshow("Src", src);
//
//
//	//计算直方图
//	int hist_size = 256;
//	float range[] = { 0,256 };
//	const float* ranges = { range };
//	cv::Mat b_hist, g_hist, r_hist;
//	cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &hist_size, &ranges, true, false);
//	cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &hist_size, &ranges, true, false);
//	cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &hist_size, &ranges, true, false);
//
//
//	//归一化
//	int hint_h = 400;
//	int hist_w = 512;
//	int bin_w = hist_w / hist_size;
//
//	cv::Mat hist_image(hist_w, hint_h, CV_8UC3, cv::Scalar(0, 0, 0));
//	cv::normalize(b_hist, b_hist, 0, hint_h, cv::NORM_MINMAX);
//	cv::normalize(g_hist, g_hist, 0, hint_h, cv::NORM_MINMAX);
//	cv::normalize(r_hist, r_hist, 0, hint_h, cv::NORM_MINMAX);
//
//	//render histogram chart
//	for (int i = 1; i < hist_size; i++) {
//		cv::line(hist_image, cv::Point((i - 1) * bin_w, hint_h - cvRound(b_hist.at<float>(i - 1))), 
//			cv::Point((i) * bin_w, hint_h - cvRound(b_hist.at<float>(i))), cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
//
//		cv::line(hist_image, cv::Point((i - 1) * bin_w, hint_h - cvRound(g_hist.at<float>(i - 1))),
//			cv::Point((i) * bin_w, hint_h - cvRound(g_hist.at<float>(i))), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
//
//		cv::line(hist_image, cv::Point((i - 1) * bin_w, hint_h - cvRound(r_hist.at<float>(i - 1))),
//			cv::Point((i) * bin_w, hint_h - cvRound(r_hist.at<float>(i))), cv::Scalar(0,0, 255), 2, cv::LINE_AA);
//	}
//
//	cv::imshow("33", hist_image);
//	cv::waitKey();
//	return 0;
//}

//直方图比较
//int main()
//{
//	cv::Mat me =  cv::imread("Me.png");
//	cv::Mat me_small_gray = cv::imread("Me_base_small.png");
//	cv::Mat me2 = cv::imread("Me1.png");
//	cv::Mat other = cv::imread("1.jpg");
//
//	//转化为hsv图像
//	cv::cvtColor(me, me, CV_BGR2HSV);
//	cv::cvtColor(me_small_gray, me_small_gray, CV_BGR2HSV);
//	cv::cvtColor(me2, me2, CV_BGR2HSV);
//	cv::cvtColor(other, other, CV_BGR2HSV);
//
//	cv::imshow("me", me);
//	cv::imshow("me_small_gray", me_small_gray);
//	cv::imshow("me2", me2);
//	cv::imshow("other", other);
//
//	auto temp_channel_count = me.channels();
//	auto data_size = me.type();
//
//	int channels[] = { 0,1 };
//	int hbins = 50, sbins = 56;//色调量化为50级 饱和度量化为56级
//	int hsize[] = { hbins,sbins };
//
//	float hranges[] = { 0, 180 };
//	float sranges[] = { 0, 256 };
//	const float* ranges[] = { hranges, sranges };
//
//	cv::MatND hist_me,hist_me_small_gray,hist_me2,hist_other;
//	cv::calcHist(&me, 1, channels, cv::Mat(), hist_me, 2, hsize, ranges, true, false);
//	cv::calcHist(&me_small_gray, 1, channels, cv::Mat(), hist_me_small_gray, 2, hsize, ranges, true, false);
//	cv::calcHist(&me2, 1, channels, cv::Mat(), hist_me2, 2, hsize, ranges, true, false);
//	cv::calcHist(&other, 1, channels, cv::Mat(), hist_other, 2, hsize, ranges, true, false);
//
//	cv::normalize(hist_me, hist_me, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());
//	cv::normalize(hist_me_small_gray, hist_me_small_gray, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());
//	cv::normalize(hist_me2, hist_me2, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());
//	cv::normalize(hist_other, hist_other, 1.0, 0.0, cv::NORM_MINMAX, -1, cv::Mat());
//
//
//	auto me_vs_me_small_gray =  cv::compareHist(hist_me, hist_me_small_gray, cv::HISTCMP_CORREL);
//	auto me_vs_me2 = cv::compareHist(hist_me, hist_me2, cv::HISTCMP_CORREL);
//	auto small_gray_vs_me2 = cv::compareHist(hist_me_small_gray, hist_me2, cv::HISTCMP_CORREL);
//
//	std::cout << "me_vs_me_small_gray:" << me_vs_me_small_gray << std::endl;
//	std::cout << "me_vs_me2:" << me_vs_me2 << std::endl;
//	std::cout << "small_gray_vs_me2:" << small_gray_vs_me2 << std::endl;
//
//	auto othre1 = cv::compareHist(hist_me, hist_other, cv::HISTCMP_CORREL);
//	auto othre2 = cv::compareHist(hist_me_small_gray, hist_other, cv::HISTCMP_CORREL);
//	auto othre3 = cv::compareHist(hist_me2, hist_other, cv::HISTCMP_CORREL);
//
//	std::cout << "othre1:" << othre1 << std::endl;
//	std::cout << "othre2:" << othre2 << std::endl;
//	std::cout << "othre3:" << othre3 << std::endl;
//
//	cv::waitKey();
//
//	return -1;
//}

//直方图反射投影
int bins = 12;
void Hist_And_Backprojection(int, void*);
cv::Mat hue;
int main()
{
	cv::Mat src1 = cv::imread("t1.jpg");

	cv::Mat src1_hsv;
	cv::cvtColor(src1, src1_hsv, CV_BGR2HSV);

	cv::imshow("src1_hsv", src1_hsv);

	//其中一个通道给分出来
	hue.create(src1_hsv.size(), src1_hsv.depth());
	int nchannels[] = { 0,0 };
	cv::mixChannels(&src1_hsv, 1, &hue, 1, nchannels, 1);

	cv::createTrackbar("Hist Bins", "src1_hsv", &bins, 180, Hist_And_Backprojection);
	Hist_And_Backprojection(0,0);
	
	cv::waitKey();
	return -1;
}
void Hist_And_Backprojection(int, void*)
{
	float range[] = { 0,180 };
	const float* histRanges = { range };
	cv::Mat hist;
	cv::calcHist(&hue, 1, 0, cv::Mat(), hist,1, &bins, &histRanges, true, false);
	cv::normalize(hist, hist, 0, 255, cv::NORM_MINMAX, -1, cv::Mat());

	cv::Mat back_prj_image;
	cv::calcBackProject(&hue, 1, 0, hist, back_prj_image, &histRanges, 1, true);
	cv::imshow("BackProj", back_prj_image);
}