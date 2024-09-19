#define M_PI 3.14159f

// INCLUDES
#include <iostream>
#include <math.h>
#include <complex>
#include <random>
#include <omp.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>
using Clock = std::chrono::high_resolution_clock;


// NAMESPACES
using namespace cv;
using namespace std;


// FUNCTIONS
// General

void complex_add(float n1_r, float n1_i, float n2_r, float n2_i, float* pResult);
void complex_mul(float n1_r, float n1_i, float n2_r, float n2_i, float* pResult);
void normalise(Mat* pData, bool forward, bool spectrum);
void shiftSpectrum(Mat* pX, Mat* pXShift);
unsigned int reverseBits(unsigned int num, int numBits);
static void createButterflyTexture(Mat* pButterfly, bool forward);



// Discrete Fourier Transforms

// optimised 2D fast fourier transform that uses a texture of pre-computed twiddle factors and input indices (fast) -- O(nlog(n))
// ! - Input signal will be altered, will become unusable after running the FFT
void fft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum);
void fft2DWrapped(Mat* pSignal, Mat* pX, bool forward, bool spectrum, Mat* pButterfly);
void horizontalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly);
void verticalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly);

// my 2D fast fourier transform implementation that involved converting the recursive definition into an interative algorithm (fast) -- O(nlog(n))
// ! - Input signal will be altered, will become unusable after running the FFT
void fft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum);
void horizontalButterfly_(int n, int row, Mat* pPingpong0, Mat* pPingpong1, bool forward);
void verticalButterfly_(int n, int col, Mat* pPingpong0, Mat* pPingpong1, bool forward);

// 2D fourier transform broken down into a series of 1D FTs on the rows followed by 1D FTs on the resultant columns (slow) -- O(n^2)
void ft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum);
void ft1D(Mat* pSignal, Mat* pX, bool forward, bool horizontal, int row_col);

// default 2D fourier transform implemented directly from textbook definition (extremely slow) -- O(n^2)
void ft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum);



// Oceanographic Spectra

// rotate spectrum across the complex plane to move simulation through time
void createHk(Mat* pH0k, Mat* pH0minusk, Mat* pHk, float t);
void createH0(Mat* pNoise, Mat* pH0k, Mat* pH0minusk);
float phillips(float K1, float K2, float k);
void createNoiseTexture(Mat* pNoise);



// Load Image From Disk
void loadImage(Mat* pImg, string image_path);



// Convert/Display Images

void im1CTo4C(Mat* pImage1C, Mat* pImage4C);
void im4CTo1C(Mat* pImage4C, Mat* pImage1C);
void imshow1CTo4C(string name, Mat* pImage1C, Mat* pImage4C);
void imshow4CTo1C(string name, Mat* pImage4C, Mat* pImage1C);
void imshow1CTo4CNew(string name, Mat* pImage1C);
void imshow4CTo1CNew(string name, Mat* pImage4C);



// Transform Types
const bool FORWARD = 1;
const bool INVERSE = 0;

const bool SPECTRUM_IMG = 1;
const bool NORMAL_IMG = 0;


// Phillips Spectrum Constants
const float L = 1000.0f; //length scale (m)
const float V = 40.0f; //wind speed
const float g = 9.80665f; //gravity
const float L_ = V*V/g;

// wind direction
const float W1 = 1.0f;
const float W2 = 1.0f;
const float w = sqrtf(W1*W1 + W2*W2); //used to normalize wind vector if (W1, W2) is not a unit-vector
const float W1_N = (w < 0.0001f) ? 0.0f : W1/w;
const float W2_N = (w < 0.0001f) ? 0.0f : W2/w;

const float A = 4.0f; //wave amplitude
const float l = 0.5f; //small wave suppression coefficient
const float directionExp = 2.0f; //exponent for suppressing waves perpendicular to wind


// Simulation Speed
const float SimulationSpeed = 10.0f;


// Texture Constants
#define M 256   // texture size (needs to be a power of 2 when using the FFTs)
constexpr int cx_log2(float n) { 
    if (n <= 2) {
        return 1;
    }
    else {
        return cx_log2(n/2)+1;
    }
}
#define NumBits cx_log2(M)


// Global Fourier Transform Variables
Mat Pingpong(M, M, CV_32FC4);       // stores intermediate results of ft calculations
Mat ButterflyForward(M, NumBits, CV_32FC4);
Mat ButterflyInverse(M, NumBits, CV_32FC4);

// Pre-compute butterfly textures
void InitButterflyTextures() {
    createButterflyTexture(&ButterflyForward, true);
    createButterflyTexture(&ButterflyInverse, false);
}


int main(){
    InitButterflyTextures();
    // display inverse butterfly texture
    //Mat butterflyInverse_(M, NumBits, CV_32FC4);
    //memcpy(butterflyInverse_.data, ButterflyInverse.data, M*NumBits*4 * sizeof(float));
    //resize(butterflyInverse_, butterflyInverse_, { M, M }, 0, 0, cv::INTER_NEAREST);
    //imshow("butterflyInverse", butterflyInverse_);



    
    // SPECTRUM IMAGE
    // noise texture
    Mat noise(M, M, CV_32FC4);
    createNoiseTexture(&noise);
    //imshow("noise", noise);
    
    // spectrum texture init
    Mat h0k(M, M, CV_32FC4);
    Mat h0minusk(M, M, CV_32FC4);
    createH0(&noise, &h0k, &h0minusk);
    //imshow("h0k", h0k);
    //imshow("h0minusk", h0minusk);
    
    
    // spectrum texture(t)
    Mat hk(M, M, CV_32FC4);
    Mat heightmap4C(M, M, CV_32FC4);
    Mat heightmap1C(M, M, CV_32FC1);
    

    // time
    chrono::steady_clock::time_point start = Clock::now();
    chrono::steady_clock::time_point current;
    float elapsedSimulationTime = 0;

    chrono::steady_clock::time_point frameStart = Clock::now();
    float frameTime = 0;
    

    // render loop
    while (true) {
        // hk
        createHk(&h0k, &h0minusk, &hk, elapsedSimulationTime);
        //imshow("hk", hk);
    
        // IFT(hk) = height map
        fft2D(&hk, &heightmap4C, INVERSE, SPECTRUM_IMG);  // can cycle between fft2D, fft2D_, ft2D, ft2D_
        //imshow("Height Map", heightmap4C);
        imshow4CTo1C("Height Map", &heightmap4C, &heightmap1C);
        //imshow4CTo1CNew("Height Map", &heightmap4C);
    

        // FT(IFT(hk)) = hk
        //Mat hk_shifted(M, M, CV_32FC4);
        //Mat hk_(M, M, CV_32FC4);
        //fft2D(&heightmap4C, &hk_shifted, FORWARD, SPECTRUM_IMG);
        //shiftSpectrum(&hk_shifted, &hk_);
        //imshow("hk_", hk_);

    
        // refresh window(s)
        int key = waitKey(1);
        if (key == 'q') {
            break;
        }
    
        // advance time
        current = Clock::now();                                                                        //to seconds
        elapsedSimulationTime = (chrono::duration_cast<chrono::nanoseconds>(current - start).count() * 0.000000001f) * SimulationSpeed;

        // calculate frametime                                                                 //to milliseconds
        frameTime = chrono::duration_cast<chrono::nanoseconds>(current - frameStart).count() * 0.000001f;
        frameStart = current;
        cout << frameTime << "ms" << endl;
    }
    
    

    /*
    // NORMAL IMAGE
    // load image
    Mat img;
    loadImage(&img, "./images/lena_gray_256.tif");
    //imshow("Image", img);
    
    // 1Channel -> 4Channel
    Mat signal(M, M, CV_32FC4);
    im1CTo4C(&img, &signal);
    //imshow("signal", signal);
    
    
    // FT(signal)
    Mat X(M, M, CV_32FC4);
    fft2D(&signal, &X, FORWARD, NORMAL_IMG);
    //X *= 255;
    //imshow("X", X);

    //imshow("signal after FFT", signal);   // confirming that ffts alter original signal
    
    // Shift
    Mat X_shifted(M, M, CV_32FC4);
    shiftSpectrum(&X, &X_shifted);
    X_shifted *= 255;
    imshow("X Shift", X_shifted);
    
    
    //// IFT(FT(signal)) = signal
    ////X /= 255;
    //Mat x(M, M, CV_32FC4);
    //fft2D(&X, &x, INVERSE, NORMAL_IMG);
    //imshow4CTo1CNew("x", &x);


    // keep window(s) open
    waitKey(0);
    */


    return 0;
}



// General

// add two complex numbers
void complex_add(float n1_r, float n1_i, float n2_r, float n2_i, float* pResult) {
    pResult[0] = n1_r + n2_r;
    pResult[1] = n1_i + n2_i;
}

// multiply two complex numbers
void complex_mul(float n1_r, float n1_i, float n2_r, float n2_i, float* pResult) {
    pResult[0] = n1_r * n2_r - n1_i * n2_i;
    pResult[1] = n1_r * n2_i + n1_i * n2_r;
}

// divides output signal by its size
void normalise(Mat* pData, bool forward, bool spectrum) {
    if (spectrum) {
        if (forward) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    float* pixel = pData->ptr<float>(y, x);

                    pixel[0] = 0;
                    pixel[3] = 1;
                }
            }
        }
        else {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    float perm = 1.0;
                    int idx = int(fmodf((float)(y + x), 2.0f));
                    perm = idx ? 1.0f : -1.0f;

                    float* pixel = pData->ptr<float>(y, x);
                    pixel[2] = (pixel[2]/(M*M))*perm;
                    pixel[1] = 0;
                    pixel[0] = 0;
                    pixel[3] = 1;
                }
            }
        }
    }

    else {
        if (forward) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    float* pixel = pData->ptr<float>(y, x);

                    pixel[2] = pixel[2]/(M*M);
                    pixel[1] = pixel[1]/(M*M);
                    pixel[0] = 0;
                    pixel[3] = 1;
                }
            }
        }
        else {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    float* pixel = pData->ptr<float>(y, x);

                    pixel[1] = 0;
                    pixel[0] = 0;
                    pixel[3] = 1;
                }
            }
        }
    }
}

// shifts spectrum from image corners to center and vice-versa by swapping quadrants
void shiftSpectrum(Mat* pX, Mat* pXShift) {
    for (int v = 0; v < M/2; v++) {
        memcpy(pXShift->ptr<float>(v + M/2, M/2), pX->ptr<float>(v), M*2 * sizeof(float));   //top-left -> bottom-right
        memcpy(pXShift->ptr<float>(v), pX->ptr<float>(v + M/2, M/2), M*2 * sizeof(float));   //bottom-right -> top-left

        memcpy(pXShift->ptr<float>(v + M/2), pX->ptr<float>(v, M/2), M*2 * sizeof(float));   //top-right -> bottom-left
        memcpy(pXShift->ptr<float>(v, M/2), pX->ptr<float>(v + M/2), M*2 * sizeof(float));   //bottom-left-> top-right
    }
}

// reverses the bits of an integer, used for re-arranging indices in the FFT functions
unsigned int reverseBits(unsigned int num, int numBits) {
    unsigned int rev = 0;
    for (int i = 0; i < numBits; i++) {
        unsigned int bit_n = (num >> i) & (unsigned int)1;
        rev = rev | (bit_n << (numBits-1 - i));
    }
    return rev;
}

// pre-computes twiddle factors and input indices for all FFT stages and stores them in a texture
static void createButterflyTexture(Mat* pButterfly, bool forward) {
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int stage = 0; stage < NumBits; stage++) {
            float k = fmodf((float)m * ((float)M/powf(2, stage + 1.0f)), M);
            float expo = 2.0f * M_PI * k / (float)M * (forward ? -1.0f : 1.0f);
            float cosExpo = cosf(expo);
            float sinExpo = sinf(expo);

            int butterflyspan = int(powf(2.0f, (float)stage));

            int butterflywing;
            if(fmodf((float)m, powf(2.0f, (float)stage + 1.0f)) < powf(2.0f, (float)stage))
                butterflywing = 1;
            else butterflywing = 0;


            float* pixel = pButterfly->ptr<float>(m, stage);
            if (stage == 0) {
                if (butterflywing == 1) {
                    pixel[2] = cosExpo; //r
                    pixel[1] = sinExpo; //g
                    pixel[0] = (float) reverseBits(m, NumBits); //b
                    pixel[3] = (float) reverseBits(m + 1, NumBits); //a
                }

                else {
                    pixel[2] = cosExpo; //r
                    pixel[1] = sinExpo; //g
                    pixel[0] = (float) reverseBits(m - 1, NumBits); //b
                    pixel[3] = (float) reverseBits(m, NumBits); //a
                }
            }
            else {
                if (butterflywing == 1) {
                    pixel[2] = cosExpo; //r
                    pixel[1] = sinExpo; //g
                    pixel[0] = (float) m; //b
                    pixel[3] = (float) m + butterflyspan; //a
                }

                else {
                    pixel[2] = cosExpo; //r
                    pixel[1] = sinExpo; //g
                    pixel[0] = (float) m - butterflyspan; //b
                    pixel[3] = (float) m; //a
                }
            }
        }
    }
}



// Discrete Fourier Transforms

// optimised 2D fast fourier transform that uses a texture of pre-computed twiddle factors and input indices
// ! - Input signal will be altered, will become unusable after running the FFT
void fft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    if (forward) {
        fft2DWrapped(pSignal, pX, forward, spectrum, &ButterflyForward);
    }
    else {
        fft2DWrapped(pSignal, pX, forward, spectrum, &ButterflyInverse);
    }
}

void fft2DWrapped(Mat* pSignal, Mat* pX, bool forward, bool spectrum, Mat* pButterfly) {
    bool pingpong = 0;
    
    // horizontal butterflies
    for (int stage = 0; stage < NumBits; stage++) {
        if (pingpong == 0) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    horizontalButterfly(stage, y, x, pSignal, &Pingpong, pButterfly);
                }
            }
        }
        else if (pingpong == 1) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    horizontalButterfly(stage, y, x, &Pingpong, pSignal, pButterfly);
                }
            }
        }

        pingpong = !pingpong;
    }

    // vertical butterflies
    for (int stage = 0; stage < NumBits; stage++) {
        if (pingpong == 0) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    verticalButterfly(stage, y, x, pSignal, &Pingpong, pButterfly);
                }
            }
        }
        else if (pingpong == 1) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    verticalButterfly(stage, y, x, &Pingpong, pSignal, pButterfly);
                }
            }
        }

        pingpong = !pingpong;
    }


    // copy data to output image
    if (pingpong == 0) {
        memcpy(pX->data, pSignal->data, M*M*4 * sizeof(float));
    }
    else {
        memcpy(pX->data, Pingpong.data, M*M*4 * sizeof(float));
    }


    // normalise
    normalise(pX, forward, spectrum);
}

void horizontalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly) {
    float data[4];
    memcpy(&data, pButterfly->ptr<float>(x, stage), 4*sizeof(float));
                                           //butterfly - blue             
    float* p_ = pPingpongIn->ptr<float>(y, (int)data[0]);
                                           //butterfly - alpha 
    float* q_ = pPingpongIn->ptr<float>(y, (int)data[3]);


    float H[2];
    //butterfly - red    - green
    complex_mul(data[2], data[1], q_[2], q_[1], H);
    complex_add(H[0], H[1], p_[2], p_[1], H);


    float* pixelPOut = pPingpongOut->ptr<float>(y, x);
    pixelPOut[2] = H[0]; //output - red
    pixelPOut[1] = H[1]; //output - green
}

void verticalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly) {
    float data[4];
    memcpy(&data, pButterfly->ptr<float>(y, stage), 4*sizeof(float));
                                        //butterfly - blue  
    float* p_ = pPingpongIn->ptr<float>((int)data[0], x);
                                        //butterfly - alpha
    float* q_ = pPingpongIn->ptr<float>((int)data[3], x);


    float H[2];
    //butterfly - red    - green
    complex_mul(data[2], data[1], q_[2], q_[1], H);
    complex_add(H[0], H[1], p_[2], p_[1], H);


    float* pixelPOut = pPingpongOut->ptr<float>(y, x);
    pixelPOut[2] = H[0]; //output - red
    pixelPOut[1] = H[1]; //output - green
}


// my 2D fast fourier transform implementation that involved converting the recursive definition into an interative algorithm
// ! - Input signal will be altered, will become unusable after running the FFT
void fft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    // horizontal butterflies
    for (int n = 2; n <= M; n*=2) {
        #pragma omp parallel for
        for (int y = 0; y < M; y++) {
            horizontalButterfly_(n, y, pSignal, &Pingpong, forward);
        }
    }

    // vertical butterflies
    for (int n = 2; n <= M; n *= 2) {
        #pragma omp parallel for
        for (int x = 0; x < M; x++) {
            verticalButterfly_(n, x, &Pingpong, pX, forward);
        }
    }


    // normalise
    normalise(pX, forward, spectrum);
}

void horizontalButterfly_(int n, int row, Mat* pPingpong0, Mat* pPingpong1, bool forward) {
    #pragma omp parallel for
    for (int v = 0; v < M; v+=n) {

        float expo = 2.0f * M_PI / n * (forward ? -1.0f : 1.0f);
        float cosExpo = cosf(expo);
        float sinExpo = sinf(expo);

        float w[2];
        w[0] = 1.0f;
        w[1] = 0.0f;


        for (int i = 0; i < n/2; i++) {
            int evenIdx = v+  2*i;
            int oddIdx =  v+  2*i + 1;
            if (n==2) {
                evenIdx = reverseBits(evenIdx, NumBits);
                oddIdx = reverseBits(oddIdx, NumBits);
            }


            float* pixelP0Odd = pPingpong0->ptr<float>(row, oddIdx);
            float* pixelP0Even = pPingpong0->ptr<float>(row, evenIdx);

            // w*odd + even;
            float butterfly_1[2];
            complex_mul(w[0], w[1], pixelP0Odd[2], pixelP0Odd[1], butterfly_1);
            complex_add(butterfly_1[0], butterfly_1[1], pixelP0Even[2], pixelP0Even[1], butterfly_1);

            // -w*odd + even;
            float butterfly_2[2];
            complex_mul(-w[0], -w[1], pixelP0Odd[2], pixelP0Odd[1], butterfly_2);
            complex_add(butterfly_2[0], butterfly_2[1], pixelP0Even[2], pixelP0Even[1], butterfly_2);


            float* pixelP1First =  pPingpong1->ptr<float>(row, v+  i);
            float* pixelP1Second = pPingpong1->ptr<float>(row, v+  i + n/2);
            pixelP1First[2] = butterfly_1[0];   pixelP1First[1] = butterfly_1[1];
            pixelP1Second[2] = butterfly_2[0];  pixelP1Second[1] = butterfly_2[1];


            // w *= wn;
            complex_mul(w[0], w[1], cosExpo, sinExpo, w);
        }
    }

    // even-odd merge step
    if (n < M) {
        #pragma omp parallel for collapse(2)
        for (int v = 0; v < M; v+=2*n) {
            for (int i = 0; i < 2*n; i+=2) {
                memcpy(pPingpong0->ptr<float>(row, v + i),   pPingpong1->ptr<float>(row, v + i/2),     4*sizeof(float));
                memcpy(pPingpong0->ptr<float>(row, v + i+1), pPingpong1->ptr<float>(row, v + i/2 + n), 4*sizeof(float));
            }
        }
    }
}

void verticalButterfly_(int n, int col, Mat* pPingpong0, Mat* pPingpong1, bool forward) {
    #pragma omp parallel for
    for (int v = 0; v < M; v+=n) {

        float expo = 2.0f * M_PI / n * (forward ? -1.0f : 1.0f);
        float cosExpo = cosf(expo);
        float sinExpo = sinf(expo);

        float w[2];
        w[0] = 1.0f;
        w[1] = 0.0f;


        for (int i = 0; i < n/2; i++) {
            int evenIdx = v+  2*i;
            int oddIdx =  v+  2*i + 1;
            if (n==2) {
                evenIdx = reverseBits(evenIdx, NumBits);
                oddIdx = reverseBits(oddIdx, NumBits);
            }


            float* pixelP0Odd = pPingpong0->ptr<float>(oddIdx, col);
            float* pixelP0Even = pPingpong0->ptr<float>(evenIdx, col);

            // w*odd + even;
            float butterfly_1[2];
            complex_mul(w[0], w[1], pixelP0Odd[2], pixelP0Odd[1], butterfly_1);
            complex_add(butterfly_1[0], butterfly_1[1], pixelP0Even[2], pixelP0Even[1], butterfly_1);

            // -w*odd + even;
            float butterfly_2[2];
            complex_mul(-w[0], -w[1], pixelP0Odd[2], pixelP0Odd[1], butterfly_2);
            complex_add(butterfly_2[0], butterfly_2[1], pixelP0Even[2], pixelP0Even[1], butterfly_2);

            
            float* pixelP1First = pPingpong1->ptr<float>(v+  i, col);
            float* pixelP1Second = pPingpong1->ptr<float>(v+  i + n/2, col);
            pixelP1First[2] = butterfly_1[0];   pixelP1First[1] = butterfly_1[1];
            pixelP1Second[2] = butterfly_2[0];  pixelP1Second[1] = butterfly_2[1];


            // w *= wn;
            complex_mul(w[0], w[1], cosExpo, sinExpo, w);
        }
    }

    // even-odd merge step
    if (n < M) {
        #pragma omp parallel for collapse(2)
        for (int v = 0; v < M; v+=2*n) {
            for (int i = 0; i < 2*n; i+=2) {
                memcpy(pPingpong0->ptr<float>(v + i, col),   pPingpong1->ptr<float>(v + i/2, col),     4*sizeof(float));
                memcpy(pPingpong0->ptr<float>(v + i+1, col), pPingpong1->ptr<float>(v + i/2 + n, col), 4*sizeof(float));
            }
        }
    }
}


// 2D fourier transform broken down into a series of 1D FTs on the rows followed by 1D FTs on the resultant columns
void ft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    // Row FTs
    #pragma omp parallel for
    for (int y = 0; y < M; y++) {
        ft1D(pSignal, &Pingpong, forward, true, y);
    }

    // Column FTs
    #pragma omp parallel for
    for (int x = 0; x < M; x++) {
        ft1D(&Pingpong, pX, forward, false, x);
    }


    // normalise
    normalise(pX, forward, spectrum);
}

// 1D fourier transform on a single row/column
void ft1D(Mat* pSignal, Mat* pX, bool forward, bool horizontal, int row_col) {
    float factor = forward ? -1.0f : 1.0f;

    if (horizontal) {
        #pragma omp parallel for
        for (int k = 0; k < M; k++) {
            float sum_real = 0;
            float sum_imag = 0;

            //#pragma omp parallel for reduction(+:sum_real) reduction(+:sum_imag)   //BROKEN
            for (int n = 0; n < M; n++) {
                float expo = ((2.0f * M_PI) / M) * k * n;
                float real =          cosf(expo);
                float imag = factor * sinf(expo);

                float* pixel = pSignal->ptr<float>(row_col, n);

                float val[2];
                complex_mul(pixel[2], pixel[1], real, imag, val);
                sum_real += val[0];
                sum_imag += val[1];
            }

            float* pixel = pX->ptr<float>(row_col, k);
            pixel[2] = sum_real;
            pixel[1] = sum_imag;
        }
    }
    else {
        #pragma omp parallel for
        for (int k = 0; k < M; k++) {
            float sum_real = 0;
            float sum_imag = 0;

            //#pragma omp parallel for reduction(+:sum_real) reduction(+:sum_imag)  //BROKEN
            for (int n = 0; n < M; n++) {
                float expo = ((2.0f * M_PI) / M) * k * n;
                float real =          cosf(expo);
                float imag = factor * sinf(expo);

                float* pixel = pSignal->ptr<float>(n, row_col);

                float val[2];
                complex_mul(pixel[2], pixel[1], real, imag, val);
                sum_real += val[0];
                sum_imag += val[1];
            }

            float* pixel = pX->ptr<float>(k, row_col);
            pixel[2] = sum_real;
            pixel[1] = sum_imag;
        }
    }
}


// default 2D fourier transform implemented directly from textbook definition
void ft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    float factor = (forward ? -1.0f : 1.0f);
    #pragma omp parallel for collapse(2)
    for (int v = 0; v < M; v++) {
        for (int u = 0; u < M; u++) {
            float sum_real = 0;
            float sum_imag = 0;

            #pragma omp parallel for collapse(2) reduction(+:sum_real) reduction(+:sum_imag)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    float expo = (2.0f * M_PI) * (x*u + y*v) / M;
                    float real =          cosf(expo);
                    float imag = factor * sinf(expo);

                    float* pixel = pSignal->ptr<float>(y, x);

                    float val[2];
                    complex_mul(pixel[2], pixel[1], real, imag, val);
                    sum_real += val[0];
                    sum_imag += val[1];

                    //complex<float> val = complex<float>(pixel[2], pixel[1]) * complex<float>(real, imag);
                    //sum_real += val.real();
                    //sum_imag += val.imag();
                }
            }

            float* pixel = pX->ptr<float>(v, u);
            pixel[2] = sum_real;
            pixel[1] = sum_imag;
        }
    }

    normalise(pX, forward, spectrum);
}



// Oceanographic Spectra

// rotate spectrum across the complex plane to move simulation through time
void createHk(Mat* pH0k, Mat* pH0minusk, Mat* pHk, float t) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float K1 = 2.0f * M_PI * (x - M/2) / L;
            float K2 = 2.0f * M_PI * (y - M/2) / L;
            float k = sqrtf(K1*K1 + K2*K2);

            float omega = sqrtf(g * k);
            float expo = omega * t;
            float cosExpo = cosf(expo);
            float sinExpo = sinf(expo);

            float* pixelH0k = pH0k->ptr<float>(y, x);
            float* pixelH0minusk = pH0minusk->ptr<float>(y, x);


            // h0k_val * <cosExpo, sinExpo>  +  h0minusk_val * <cosExpo, -sinExpo>;
            float a[2];
            float b[2];
            float hk_val[2];
            complex_mul(pixelH0k[2], pixelH0k[1], cosExpo,  sinExpo, a);
            complex_mul(pixelH0minusk[2], pixelH0minusk[1], cosExpo, -sinExpo, b);
            complex_add(a[0], a[1], b[0], b[1], hk_val);


            float* pixelHk = pHk->ptr<float>(y, x);
            pixelHk[2] = hk_val[0];
            pixelHk[1] = hk_val[1];
            pixelHk[0] = 0;
            pixelHk[3] = 1;
        }
    }
}

// create initial spectrum
void createH0(Mat* pNoise, Mat* pH0k, Mat* pH0minusk) {
    const float bound = 4000.0f;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float K1 = 2.0f * M_PI * (x - M/2) / L;
            float K2 = 2.0f * M_PI * (y - M/2) / L;
            float k = sqrtf(K1*K1 + K2*K2);
            if (k < 0.0001f) {
                k = 0.0001f;
            }
            
            float suppressionFactor = expf(-k*k*l*l);
            

            float* pixelNoise = pNoise->ptr<float>(y, x);

            float h0k_val = max(-bound, min(bound, sqrtf((suppressionFactor * phillips(K1, K2, k)) * 0.5f)));
            float* pixelH0k = pH0k->ptr<float>(y, x);
            pixelH0k[2] = pixelNoise[2] * h0k_val;
            pixelH0k[1] = pixelNoise[1] * h0k_val;

            float h0minusk_val = max(-bound, min(bound, sqrtf((suppressionFactor * phillips(-K1, -K2, k)) * 0.5f)));
            float* pixelH0minusk = pH0minusk->ptr<float>(y, x);
            pixelH0minusk[2] = pixelNoise[0] * h0minusk_val;
            pixelH0minusk[1] = pixelNoise[3] * h0minusk_val;
        }
    }
}

// Phillips Spectrum equation
float phillips(float K1, float K2, float k) {
    float K1_N = K1/k;
    float K2_N = K2/k;

    return A * (expf(-powf(k*L_, -2)) * powf(k, -4)) * powf(abs(K1_N*W1_N + K2_N*W2_N), directionExp);
}

// generate a texture of pseudorandom noise
void createNoiseTexture(Mat* pNoise) {
    float lower_bound = -1;
    float upper_bound = 1;
    uniform_real_distribution<float> unif(lower_bound, upper_bound);
    default_random_engine re;


    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float* pixel = pNoise->ptr<float>(y, x);

            pixel[2] = unif(re);
            pixel[1] = unif(re);
            pixel[0] = unif(re);
            pixel[3] = unif(re);
        }
    }
}



// Load Image From Disk
void loadImage(Mat* pImg, string image_path) {
    *pImg = imread(image_path, IMREAD_GRAYSCALE);
    pImg->convertTo(*pImg, CV_32FC1);
    *pImg /= 255;
    resize(*pImg, *pImg, { M, M }, 0, 0, cv::INTER_NEAREST);
}



// Convert/Display Images

// grayscale to BGRA
void im1CTo4C(Mat* pImage1C, Mat* pImage4C) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float* pixel1C = pImage1C->ptr<float>(y, x);
            float* pixel4C = pImage4C->ptr<float>(y, x);
            pixel4C[2] = *pixel1C;
            pixel4C[1] = *pixel1C;
            pixel4C[0] = *pixel1C;
            pixel4C[3] = 1;
        }
    }
}

// BGRA to grayscale
void im4CTo1C(Mat* pImage4C, Mat* pImage1C) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float* pixel4C = pImage4C->ptr<float>(y, x);
            float* pixel1C = pImage1C->ptr<float>(y, x);
            *pixel1C = pixel4C[2];
        }
    }
}

void imshow1CTo4C(string name, Mat* pImage1C, Mat* pImage4C) {
    im1CTo4C(pImage1C, pImage4C);
    imshow(name, *pImage4C);
}

void imshow4CTo1C(string name, Mat* pImage4C, Mat* pImage1C){
    im4CTo1C(pImage4C, pImage1C);
    imshow(name, *pImage1C);
}

void imshow1CTo4CNew(string name, Mat* pImage1C) {
    Mat image_4C(M, M, CV_32FC4);
    im1CTo4C(pImage1C, &image_4C);

    imshow(name, image_4C);
}

void imshow4CTo1CNew(string name, Mat* pImage4C){
    Mat image_1C(M, M, CV_32FC1);
    im4CTo1C(pImage4C, &image_1C);

    imshow(name, image_1C);
}