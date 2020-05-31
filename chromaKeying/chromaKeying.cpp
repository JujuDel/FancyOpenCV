#include <iostream>
#include <vector>
#include <string>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// ########## CONST ##########

const char    *g_windowImg{ "Chroma Keying" };
const char    *g_windowParam{ "Parameters" };
const int      g_fps{ 20 };
const cv::Size g_size{ 900, 600 };

// ########## PARAMETERS ##########

//
bool g_isVideoPlaying{ false };
bool g_isSamplingFrame{ true };

// For dragging
bool g_dragging = false;
cv::Point g_corner1, g_corner2;

// Images
cv::Mat g_imgFg, g_imgBg;  
  
// Array of vectors of pairs min/max for BGR colors
std::vector<std::pair<double, double>> g_BGR[3];
bool goBack()
{
	if (g_BGR[0].empty()) return false;
	
	g_BGR[0].pop_back();
	g_BGR[1].pop_back();
	g_BGR[2].pop_back();
	return true;
}

// Tolerance
bool              g_isToleranceUp{ true };
int               g_tolerance{ 0 };
std::string const g_buttonText{ "Click here to change tolerance" };
cv::Mat3b         g_button;
#define BUTTONTEXT (g_isToleranceUp ? "Tolerance Up" : "Tolerance Down") //! Adaptative text

// Color Cast
int g_colorCast{ 0 };

// Softness
int g_softness{ 0 };

// ########## UTILITIES ##########

// Do min/max with different types
template<typename T1, typename T2>
T1 myMin(const T1 &a, const T2 &b) { return a < b ? a : b; }
template<typename T1, typename T2>
T1 myMax(const T1 &a, const T2 &b) { return a > b ? a : b; }

// Display pair content
template<typename T1, typename T2>
std::ostream & operator << (std::ostream &out, const std::pair<T1, T2> &p) {
	out << "(Min: " << p.first << "; Max: " << p.second << ")";
	return out;
}

std::string myLower(std::string text) {
	std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c){ return std::tolower(c); });
	return text;
}

// ########## FORWARD DECLARATIONS ##########

// Callbacks
void onMouse(int event, int x, int y, int flags, void* userdata);
void onButton(int event, int x, int y, int flags, void* userdata);
void onTolerance(int, void*);
void onSoftness(int, void*);
void onColorCast(int, void*);

// Displays
void header();
void writeTextButton();
void drawOverlayMessage(cv::Mat &img, const char *text);
void printBGR();

// Apply Chroma Keying
void chromaKeying(const cv::Mat &bg, const cv::Mat &fg, cv::Mat &dst);

// Sample the current frame
void samplingFrame(const cv::Mat &img, char &keyPressed);

// ########## MAIN ##########

int main()
{
  // Display some text
  header();
  
  // Read the videos
  cv::VideoCapture videoFg("greenscreen-demo.mp4");
  cv::VideoCapture videoBg("newBackground.mp4");
  
  // Check if the videos opened successfully
  if ( !videoFg.isOpened() || !videoBg.isOpened() )
  {
    std::cout << "/!\\ Error opening video files /!\\" << std::endl;
    return -1;
  }
  
  // Param's window
  cv::namedWindow(g_windowParam);

  // Trackabars  
  cv::createTrackbar("Tolerance", g_windowParam, &g_tolerance, 100, onTolerance);
  cv::createTrackbar("Softness", g_windowParam, &g_softness, 100, onSoftness);
  cv::createTrackbar("Color cast", g_windowParam, &g_colorCast, 50, onColorCast);
  
  // Mouse callback
  cv::setMouseCallback(g_windowParam, onButton);
  writeTextButton();
  cv::imshow(g_windowParam, g_button);

  // Main window
  cv::namedWindow(g_windowImg);
  
  // Mouse callback
  cv::setMouseCallback(g_windowImg, onMouse);

  char keyPressed;
  while (keyPressed != 27)
  {
	// Capture frame-by-frame
    videoFg >> g_imgFg;
	videoBg >> g_imgBg;
	
	// If the frame is empty, break immediately
	if (g_imgFg.empty() || g_imgBg.empty()) break;
	
	// Resize the frames
	cv::resize(g_imgFg, g_imgFg, g_size);
	cv::resize(g_imgBg, g_imgBg, g_size);
	
	// Apply chromaKeying
	cv::Mat dst;
	chromaKeying(g_imgBg, g_imgFg, dst);
	
	if (!g_isVideoPlaying)
	{
		// Sample the current frame
		samplingFrame(dst, keyPressed);
	}
	else
	{
		// Draw the rectangle
		if (g_dragging) cv::rectangle(dst, g_corner1, g_corner2, cv::Scalar(255,255,0), 2, cv::LINE_AA);
		// Display the frame
		drawOverlayMessage( dst, "Press SPACE to pause the video" );
		cv::imshow( g_windowImg, dst );
		keyPressed = (char) cv::waitKey(g_fps);
	}
	
	// Restart videos
	if (keyPressed == 'R' || keyPressed == 'r')
	{
		videoFg.release();
		videoBg.release();
		
		videoFg = cv::VideoCapture("greenscreen-demo.mp4");
		videoBg = cv::VideoCapture("greenscreen-asteroid.mp4");
		
		std::cout << "Videos restarted" << std::endl;
	}
	// Z
	if (keyPressed == 'Z' || keyPressed == 'z')
		if (goBack())
			std::cout << "Color sampling cancelled" << std::endl;
		else
			std::cout << "Nothing to cancel" << std::endl;
	// Spacebar
	if (keyPressed == 32)
	{
		// Pause/Play the video
		g_isVideoPlaying = !g_isVideoPlaying;
		g_isSamplingFrame = !g_isVideoPlaying;
		
		std::cout << (g_isVideoPlaying ? "Play" : "Pause") << std::endl;
	}
	// Print values
	if (keyPressed == 'P' || keyPressed == 'p')
	{
		std::cout << "Values:" << std::endl;
		printBGR();
	}
  }
  
  // When everything done, release the video capture object
  videoFg.release();
  videoBg.release();

  if (keyPressed != 27)
  {
    std::string ans;
    std::cout << "End of video reached. Restart?" << std::endl;
    std::cout << "(Y/N)? ";
    std::getline(std::cin, ans);
  
    ans = myLower(ans);
    if (ans == "y" || ans == "yes" || ans == "o") main();
  }
  
  // Closes all the frames
  cv::destroyAllWindows();
  
  return 0;
}

// ########## CALLBACKS ##########

void onMouse(int action, int x, int y, int flags, void* userdata)
{	
	if ( action == cv::EVENT_LBUTTONDOWN )
	{
		// Register the selected point
		g_corner1 = cv::Point(x,y);
		g_corner2 = g_corner1;
		
		// Allow dragging
		g_dragging = true;
	}
	else if ( action == cv::EVENT_LBUTTONUP )
	{
		// Cancel dragging
		g_dragging = false;
		
		// Extract the selected region
		cv::Point l_Size{ g_corner1 - g_corner2 };
		int l_width{ std::abs(l_Size.x) };
		int l_height{ std::abs(l_Size.y) };
		
		// Update the background remover
		if (l_width < 2 || l_height < 2)
		{
			cv::Vec3b pix{ g_imgFg.at<cv::Vec3b>(g_corner1) };
			
			for (int i = 0; i < 3; ++i)
			{
				if (g_BGR[i].empty())
				{
					g_BGR[i].push_back(std::pair<double, double>(pix[i], pix[i]));
				}
				else
				{
					g_BGR[i].push_back(std::pair<double, double>(
						myMin(g_BGR[i].back().first, pix[i]),
						myMax(g_BGR[i].back().second, pix[i])
					));
				}
			}
		}
		else
		{
			int l_x{ std::min(g_corner1.x, g_corner2.x) };
			int l_y{ std::min(g_corner1.y, g_corner2.y) };
			cv::Mat ROI{ cv::Mat(g_imgFg, cv::Rect(l_x, l_y, l_width, l_height)) };
			
			// Get the 3 channels
			cv::Mat BGR[3];
			cv::split(ROI, BGR);
			
			for (int i = 0; i < 3; ++i)
			{
				// Find the global max and min in the channel
				double l_min, l_max;
				cv::minMaxLoc(BGR[i], &l_min, &l_max);
				
				// Update the background remover
				if (g_BGR[i].empty())
				{
					g_BGR[i].push_back(std::pair<double, double>(l_min, l_max));
				}
				else
				{
					g_BGR[i].push_back(std::pair<double, double>(
						std::min(g_BGR[i].back().first, l_min),
						std::max(g_BGR[i].back().second, l_max)
					));
				}
			}
		}
		std::cout << "Color sampling done" << std::endl;
		
		// Sampling is done
		g_isSamplingFrame = false;
	}
	else if ( g_dragging )
	{
		// Update the selected region
		x = std::min(g_size.width, std::max(0, x));
		y = std::min(g_size.height, std::max(0, y));
		g_corner2 = cv::Point(x,y);
	}
}

void onButton(int action, int x, int y, int flags, void* userdata)
{
	if (action == cv::EVENT_LBUTTONDOWN)
    {
		g_tolerance     = 0;
		g_isToleranceUp = !g_isToleranceUp;
		
		cv::createTrackbar("Tolerance", g_windowParam, &g_tolerance, 100, onTolerance);
		writeTextButton();
		
		cv::imshow(g_windowParam, g_button);
		
		// Call onTolerance callback
		onTolerance(0, 0);
    }
}

void onTolerance(int, void*)
{
	// To force display updates
	g_isSamplingFrame = false;
}

void onSoftness(int, void*)
{
	// To force display updates
	g_isSamplingFrame = false;
	
	if (g_softness != 0) g_softness = 100 - g_softness;

	std::cout << "Not yet implemented..." << std::endl;
}

void onColorCast(int, void*)
{
	// To force display updates
	g_isSamplingFrame = false;
}

// ########## DISPLAYS ##########

void header()
{
	std::cout << std::endl;
	std::cout << "##################################" << std::endl;
	std::cout << "#                                #" << std::endl;
	std::cout << "#       Chroma Keying Tool       #" << std::endl;
	std::cout << "#         Julien DELCLOS         #" << std::endl;
	std::cout << "#                                #" << std::endl;
	std::cout << "##################################" << std::endl;
	std::cout << std::endl;
	std::cout << "Press 'R' on the image to restart the videos." << std::endl;
	std::cout << std::endl;
	std::cout << "Press 'P' on the image to print the sampling values." << std::endl;
	std::cout << std::endl;
	std::cout << "Press 'Z' on the image to cancel the last sampling." << std::endl;
	std::cout << std::endl;
	std::cout << "Press 'SPACE' on the image to Play/Pause the video." << std::endl;
	std::cout << std::endl;
	std::cout << "Press 'ESC' on the image to quit." << std::endl;
	std::cout << std::endl;
	std::cout << "------------    LOG    ------------" << std::endl;
	std::cout << std::endl;
}

void printBGR()
{
	if (g_BGR[0].empty())
	{
		std::cout << "   no values" << std::endl;
		return;
	}
	
	std::cout << "   B: " << g_BGR[0].back() << std::endl;
	std::cout << "   G: " << g_BGR[1].back() << std::endl;
	std::cout << "   R: " << g_BGR[2].back() << std::endl;
}

void writeTextButton()
{
  g_button = cv::Mat3b( 50, 410, cv::Vec3b(125, 125, 125) );	
  
  int fontFace{ cv::FONT_HERSHEY_SIMPLEX };
  double fontScale{ 0.5 };
  int thickness{ 1 };
  
  cv::Size smallTextSize{ cv::getTextSize(g_buttonText, fontFace, fontScale, thickness, NULL) };
  cv::Size bigTextSize{ cv::getTextSize(BUTTONTEXT, fontFace, fontScale, 2, NULL) };
  
  // Center the text
  cv::Point smallTextOrg(
    (410 - smallTextSize.width)/2,
	smallTextSize.height
  );
  cv::Point bigTextOrg(
    (410 - bigTextSize.width)/2,
    50 - bigTextSize.height
  );
  
  cv::putText(g_button, g_buttonText, smallTextOrg, fontFace, fontScale, cv::Scalar(255,255,255), thickness, 8);
  cv::putText(g_button, BUTTONTEXT, bigTextOrg, fontFace, fontScale, cv::Scalar(255,255,255), 2, 8);
}

void drawOverlayMessage(cv::Mat &img, const char *text)
{
  cv::Mat overlay{ img.clone() };
  int fontFace{ cv::FONT_HERSHEY_SIMPLEX };
  double fontScale{ 1 };
  int thickness{ 2 };
  
  int baseline{ 0 };
  cv::Size textSize{ cv::getTextSize(text, fontFace, fontScale, thickness, &baseline) };
  baseline += thickness;

  // Center the text
  cv::Point textOrg(
    (img.cols - textSize.width)/2,
	30
  );

  // Draw the box
  cv::rectangle(
    overlay,
	textOrg + cv::Point(0, baseline),
	textOrg + cv::Point(textSize.width, -50),
	cv::Scalar(0,0,255),
	-1
  );
  
  // Overlay the box
  cv::addWeighted(overlay, 0.5, img, 0.5, 0, img);
  
  // Put the text
  cv::putText(img, text, textOrg, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness, 8);
}

// ########## SAMPLING ##########

void samplingFrame(const cv::Mat &img, char &keyPressed)
{
	// Enable the looping
	g_isSamplingFrame = true;
	
	while (g_isSamplingFrame && keyPressed != 27)
	{
		cv::Mat l_img{ img.clone() };
		drawOverlayMessage( l_img, "Select the background to remove" );
		if (g_dragging)
		{
			// Draw the rectangle
			cv::rectangle(l_img, g_corner1, g_corner2, cv::Scalar(255,255,0), 2, cv::LINE_AA);
		}
		
		cv::imshow( g_windowImg, l_img );
		keyPressed = (char) cv::waitKey(20);
		
		if (keyPressed == 'N' || keyPressed == 'n' ||
			keyPressed == 'Z' || keyPressed == 'z' ||
			keyPressed == 'P' || keyPressed == 'p' ||
			keyPressed == 'R' || keyPressed == 'r' ||
			keyPressed == 32) break;
	}
}

// ########## CHROMA KEYING ##########

void performSoftness(const cv::Mat &fg, cv::Mat &mask)
{
#if 0
	cv::Mat fg_gray;
	cv::cvtColor(fg, fg_gray, cv::COLOR_BGR2GRAY);
	cv::blur(fg_gray, fg_gray, cv::Size(3,3));
	
	if (g_softness != 0)
	{
		cv::Mat canny;
		cv::Canny(fg_gray, canny, g_softness, g_softness*3, 3);
		//cv::imshow("Canny", canny);
	}
#endif
	// Topic ongoing
}

void constructMask(const cv::Mat &fg, cv::Mat &mask)
{
	mask = cv::Mat(fg.rows, fg.cols, CV_8UC1);
	
	double meanG{ (g_BGR[1].back().second + g_BGR[1].back().first) / 2.0 };
	double tolG_m{ g_tolerance * meanG / 100.0 };
	double tolG_M{ g_tolerance * meanG / 100.0};

	if (g_isToleranceUp) tolG_m *= -1;
	else tolG_M *= -1;
	
	for (int y = 0; y < fg.rows; y++)
	{
		for (int x = 0; x < fg.cols; x++)
		{
			cv::Vec3b pix{ fg.at<cv::Vec3b>(y, x) };
			if (pix.val[0] >= g_BGR[0].back().first && pix.val[0] <= g_BGR[0].back().second &&
				pix.val[2] >= g_BGR[2].back().first && pix.val[2] <= g_BGR[2].back().second &&
				pix.val[1] >= g_BGR[1].back().first + tolG_m &&
				pix.val[1] <= g_BGR[1].back().second + tolG_M)
			{
				mask.at<uchar>(y,x) = 0;
			}
			else
			{
				mask.at<uchar>(y,x) = 255;
			}
		}
	}
	
	// Color casting
	cv::Mat kernel{ cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * g_colorCast + 1, 2 * g_colorCast + 1)) };
	cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
	//cv::imshow("Mask", mask);
}

void chromaKeying(const cv::Mat &bg, const cv::Mat &fg, cv::Mat &dst)
{
	assert(bg.size() == fg.size());
	
	if (g_BGR[0].empty())
	{
		dst = fg;
	}
	else
	{
		cv::Mat mask;
		constructMask(fg, mask);
		performSoftness(fg, mask);

		dst = cv::Mat(fg.rows, fg.cols, CV_8UC3);
		
		for (int y = 0; y < fg.rows; y++)
		{
			for (int x = 0; x < fg.cols; x++)
			{
				if (mask.at<uchar>(y,x) == 255) dst.at<cv::Vec3b>(y,x) = fg.at<cv::Vec3b>(y,x);
				else dst.at<cv::Vec3b>(y,x) = bg.at<cv::Vec3b>(y,x);
			}
		}
	}
}