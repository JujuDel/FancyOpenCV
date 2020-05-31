#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

// ########## PARAMETERS ##########

unsigned int         g_nbImagesSaved{ 0 };         //! Counter of saved images

const char          *g_window{ "BlemishRemoval" }; //! The name of the OpenCV window
  
cv::Mat              g_source;                     //! The image initially loaded
cv::Mat              g_image;                      //! The image updated at each clicks
std::vector<cv::Mat> g_memory;                     //! Memory of the images
int                  g_radius{ 15 };               //! The radius for patch selection on click

bool                 g_dragging{ false };          //! Whether or not the mouse is dragged

// ########## FORWARD DECLARATIONS ##########

void header();
void readImage(const char*);
bool goBack();
void callbackMouse(int, int, int, int, void*);

// ########## MAIN ##########

int main()
{
  // Display some text
  header();
  
  // Create the OpenCV window
  cv::namedWindow(g_window);
  
  readImage("blemish.png");
  
  // highgui function called when mouse events occur
  cv::setMouseCallback(g_window, callbackMouse);
  
  // loop until escape character is pressed
  int k{ 0 };
  while(k != 27)
  {
    // Display the current g_image
    cv::imshow(g_window, g_image);

    // Wait 20ms for key input
    k = cv::waitKey(20);
    if (k == 'S' || k == 's')
    {
      cv::imwrite("out_" + std::to_string(g_nbImagesSaved++) + ".png", g_image);
      std::cout << "Image saved..." << std::endl; 
    }
    else if (k == 'R' || k == 'r')
    {
      g_memory.push_back(g_image);
      g_image = g_source.clone();
      std::cout << "Image reset..." << std::endl; 
    }
    else if (k == 'Z' || k == 'z')
    {
      if ( goBack() )
        std::cout << "Action cancelled..." << std::endl; 
    }
  }
  return 0;
}

// ########## UTILITIES ##########

void header()
{
  std::cout << std::endl;
  std::cout << "##################################" << std::endl;
  std::cout << "#                                #" << std::endl;
  std::cout << "#      Blemish Removal Tool      #" << std::endl;
  std::cout << "#         Julien DELCLOS         #" << std::endl;
  std::cout << "#                                #" << std::endl;
  std::cout << "##################################" << std::endl;
  std::cout << std::endl;
  std::cout << "Left clic on the blemish to remove." << std::endl;
  std::cout << "You can repeat the process." << std::endl;
  std::cout << std::endl;
  std::cout << "Press 'S' on the image to save." << std::endl;
  std::cout << "Multiple images can be saved." << std::endl;
  std::cout << std::endl;
  std::cout << "Press 'R' on the image to reset." << std::endl;
  std::cout << std::endl;
  std::cout << "Press 'Z' on the image to cancel." << std::endl;
  std::cout << std::endl;
  std::cout << "Press 'ESC' on the image to quit." << std::endl;
  std::cout << std::endl;
  std::cout << "------------    LOG    ------------" << std::endl;
  std::cout << std::endl;
}

void readImage(const char *f_pathToImg)
{
  g_source = cv::imread(f_pathToImg, 1);
  g_image = g_source.clone();

  std::cout << "Image " << f_pathToImg << " read." << std::endl;
}

bool goBack()
{
  if ( !g_memory.empty() )
  {
    g_image = g_memory.back();
    g_memory.pop_back();
    return true;
  }
  return false;
}

// ########## PATCH SELECTION ##########

// Return the patchs by taking care of the ranges
// The patch at the idx 0 is the selected one
// Then if patchs of the same sizes can be extract, up to 4 adddictional will be addded:
//              patch
// patch - clicked patch - patch
//              patch
const std::vector<cv::Mat> getPatchs(int &x, int &y)
{
  int l_width{ g_image.cols };
  int l_height{ g_image.rows };

  std::vector<cv::Mat> l_patchs;

  // SELECTED PATCH
  if (y - g_radius < 0)
  {
    y = g_radius;
  }
  else if (y + g_radius >= l_height)
  {
    y = l_height - 1 - g_radius;
  }
  int y_min = y - g_radius;
  int y_max = y + g_radius;

  if (x - g_radius < 0)
  {
    x = g_radius;
  }
  else if (x + g_radius >= l_width)
  {
    x = l_width - 1 - g_radius;
  }
  int x_min = x - g_radius;
  int x_max = x + g_radius;

  l_patchs.push_back(
    g_image(cv::Range(y_min, y_max), cv::Range(x_min, x_max))
  );

  // VERTICAL PATCHS
  if (y_min - 2 * g_radius >= 0)
  {
    l_patchs.push_back(
      g_image(cv::Range(y_min - 2 * g_radius, y_min), cv::Range(x_min, x_max))
    );
  }
  if (y_max + 2 * g_radius < l_height)
  {
    l_patchs.push_back(
      g_image(cv::Range(y_max, y_max + 2 * g_radius), cv::Range(x_min, x_max))
    );
  }

  // HORIZONTAL PATCHS
  if (x_min - 2 * g_radius >= 0)
  {
    l_patchs.push_back(
      g_image(cv::Range(y_min, y_max),cv::Range(x_min - 2 * g_radius, x_min))
    );
  }
  if (x_max + 2 * g_radius < l_width)
  {
    l_patchs.push_back(
      g_image(cv::Range(y_min, y_max),cv::Range(x_max, x_max + 2 * g_radius))
    );
  }

  // Return
  return l_patchs;
}

// Return the idx of the patch with the lowest gradient
const unsigned int getBestSobel(const std::vector<cv::Mat> &f_patchs)
{
  unsigned int l_best;
  unsigned int l_bestSum = -1; //! At first; it'l be the highest uint possible ;-)
  
  // Sobel paramaters
  const unsigned int scale{ 1 };
  const unsigned int delta{ 0 };
  const unsigned int ddepth{ CV_16S };

  for (unsigned int idx = 0; idx < f_patchs.size(); ++idx)
  {
    // Get the patch
    cv::Mat l_patch{ f_patchs[idx] };
  
    // Convert to gray
    cv::Mat l_patchGray;
    cv::GaussianBlur( l_patch, l_patch, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    cv::cvtColor( l_patch, l_patchGray, cv::COLOR_BGR2GRAY );

    // Apply Sobel x-wisely
    cv::Mat l_grad_x;
    cv::Sobel( l_patchGray, l_grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( l_grad_x, l_grad_x );

    // Apply Sobel y-wisely
    cv::Mat l_grad_y;
    cv::Sobel( l_patchGray, l_grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( l_grad_y, l_grad_y );

    // Get the final Sobel gradient
    cv::Mat l_grad;
    cv::addWeighted( l_grad_x, 0.5, l_grad_y, 0.5, 0, l_grad );

    // Compute the score
    unsigned int l_sum{ static_cast<unsigned int>(cv::sum(l_grad)[0]) };

    // Compare the score
    if (l_sum < l_bestSum)
    {
      l_bestSum = l_sum;
      l_best = idx;
    }
  }
  // Return
  return l_best;
 }

void blemishRemoval(int x, int y)
{
  // Get the current patch and the patchs around
  const std::vector<cv::Mat> l_patchs{ getPatchs( x, y ) };

  // Get the patch with the less gradient
  const unsigned int idxBest{ getBestSobel(l_patchs) };
  
  // Replace the current patch with the patch with the less gradient
  cv::Mat mask{ 255 * cv::Mat::ones(l_patchs[idxBest].rows, l_patchs[idxBest].cols, l_patchs[idxBest].depth()) };
  cv::Point center{ x, y };
  cv::seamlessClone(
    l_patchs[idxBest], // src
    g_image, // dst
    mask, //mask
    center, //center
    g_image, // output 
    cv::NORMAL_CLONE // flag
  );
}

// ########## HIGHGUI CALLBACK ##########

// Function which will be called on mouse input
void callbackMouse(int action, int x, int y, int flags, void *userdata)
{
  if ( action == cv::EVENT_LBUTTONDOWN )
  {
    // Save the current image before applying modification on it
    g_memory.push_back(g_image.clone());
    // Replace the clicked patch
    blemishRemoval(x,y);
    // The mouse can be dragged
    g_dragging = true;
  }
  else if ( action == cv::EVENT_LBUTTONUP )
  {
    // The mouse can't be dragged
    g_dragging = false;

    std::cout << "Image updated..." << std::endl;
  }
  else if ( g_dragging )
  {
    // Do not save the current image on drag to make easy the revert

    // Replace the clicked patch
    blemishRemoval(x,y);
  }
}