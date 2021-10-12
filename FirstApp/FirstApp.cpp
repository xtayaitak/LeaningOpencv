// FirstApp.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <opencv2/opencv.hpp>

//看图片
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_videoio3414d.lib")
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
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_videoio3414d.lib")
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
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_videoio3414d.lib")
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
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")


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
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")
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
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")
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

//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")
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


//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")
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




#pragma comment(lib,"opencv_core3414d.lib")
#pragma comment(lib,"opencv_highgui3414d.lib")
#pragma comment(lib,"opencv_imgcodecs3414d.lib")
#pragma comment(lib,"opencv_imgproc3414d.lib")
//形态学操作

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



#pragma comment(lib,"opencv_core3414d.lib")
#pragma comment(lib,"opencv_highgui3414d.lib")
#pragma comment(lib,"opencv_imgcodecs3414d.lib")
#pragma comment(lib,"opencv_imgproc3414d.lib")
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
//#pragma comment(lib,"opencv_highgui3414d.lib")
//#pragma comment(lib,"opencv_imgproc3414d.lib")
//#pragma comment(lib,"opencv_core3414d.lib")
//#pragma comment(lib,"opencv_imgcodecs3414d.lib")
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

#pragma comment(lib,"opencv_highgui3414d.lib")
#pragma comment(lib,"opencv_imgproc3414d.lib")
#pragma comment(lib,"opencv_core3414d.lib")
#pragma comment(lib,"opencv_imgcodecs3414d.lib")
int main()
{
	cv::Mat src = cv::imread("test.jpg");
	cv::namedWindow("Input Image", cv::WINDOW_AUTOSIZE);
	cv::imshow("Input Image", src);

	cv::Mat dst, dst2;

	cv::pyrUp(src, dst, cv::Size(src.cols * 2, src.rows * 2)); // 上采样
	cv::imshow("Output Image", dst);

	cv::pyrDown(src, dst2, cv::Size(src.cols / 2, src.rows / 2)); // 降采样
	cv::imshow("Output Image2", dst2);

	//DOG
	cv::Mat g1, g2, gray_src, dog_img;
	cv::cvtColor(src, gray_src, CV_BGR2GRAY);
	cv::GaussianBlur(gray_src, g1, cv::Size(5, 5), 0, 0);
	cv::GaussianBlur(g1, g2, cv::Size(5, 5), 0, 0);
	cv::subtract(g1, g2, dog_img);

	//归一化显示 如果不弄，会显示很暗
	cv::normalize(dog_img, dog_img, 255, 0, cv::NORM_MINMAX);
	cv::imshow("Dog Img", dog_img);
	cv::waitKey(0);
	std::cout << "Hello World!\n";
	return 0;
}