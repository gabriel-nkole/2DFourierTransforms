#define _USE_MATH_DEFINES

// Includes
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


// Namespaces
using namespace cv;
using namespace std;


// Functions
// 
// general
void normalise(Mat* pData, bool forward, bool spectrum);
void shiftSpectrum(Mat* pX, Mat* pXShift);
unsigned int reverseBits(unsigned int num, int numBits);
static void createButterflyTexture(Mat* pButterfly, bool forward);


// optimised 2D fast fourier transform that uses a texture of pre-computed twiddle factors and input indices (fast) -- O(nlog(n))
void fft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum);
void fft2DWrapped(Mat* pSignal, Mat* pX, bool forward, bool spectrum, Mat* pButterfly);
void horizontalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly);
void verticalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly);

// My 2D fast fourier transform implementation that involved converting the recursive definition into an interative one (fast) -- O(nlog(n))
void fft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum);
void horizontalButterfly_(int n, int row, Mat* pPingpong0, Mat* pPingpong1, bool forward);
void verticalButterfly_(int n, int col, Mat* pPingpong0, Mat* pPingpong1, bool forward);

// 2D fourier transform broken down as a series of 1D FTs on the rows followed by 1D FTs on the resultant columns (slow) -- O(n^2)
void ft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum);
void ft1D(Mat* pSignal, Mat* pX, bool forward, bool horizontal, int row_col);

// default 2D fourier transform implemented directly from textbook definition (extremely slow) -- O(n^2)
void ft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum);


// create oceanographic spectra
void createHk(Mat* pH0k, Mat* pH0minusk, Mat* pHk, double t);
void createH0(Mat* pNoise, Mat* pH0k, Mat* pH0minusk);
double phillips(double K1, double K2, double k);
void createNoiseTexture(Mat* pNoise);

// load image from disk
Mat loadImage(string image_path);

// convert/display images
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
const double L = 1000;
const double V = 40;
const double g = 9.80665;
const double L_ = V*V/g;

const double W1 = 1;
const double W2 = 1;
const double w = sqrt(W1*W1 + W2*W2);

const double A = 4;
const double l = 0.5;

// Simulation Speed
const double SimulationSpeed = 1.0;

// Texture Constants
#define M 256   // texture size (should be a power of 2 when using the FFTs)
constexpr int cx_log2(int n) { 
    if (n <= 2) {
        return 1;
    }
    else {
        return cx_log2(n/2)+1;
    }
}
#define BitsConst cx_log2(M)


// Global Fourier Transform Variables
Mat Pingpong(M, M, CV_64FC4);
Mat ButterflyForward(M, BitsConst, CV_64FC4);
Mat ButterflyInverse(M, BitsConst, CV_64FC4);

//pre-compute butterfly textures
void InitButterflyTextures() {
    createButterflyTexture(&ButterflyForward, true);
    createButterflyTexture(&ButterflyInverse, false);
}


int main(){
    InitButterflyTextures();
    // display butterfly texture
    //Mat butterflyCopy(M, BitsConst, CV_64FC4);
    //memcpy(butterflyCopy.data, ButterflyInverse.data, M*BitsConst*4 * sizeof(double));
    //resize(butterflyCopy, butterflyCopy, { M, M }, 0, 0, cv::INTER_NEAREST);
    //imshow("butterflyInverse", butterflyCopy);



    
    //Spectrum Image
    // noise image
    Mat noise(M, M, CV_64FC4);
    createNoiseTexture(&noise);
    //imshow("noise", noise);
    
    // spectrum texture init
    Mat h0k(M, M, CV_64FC4);
    Mat h0minusk(M, M, CV_64FC4);
    createH0(&noise, &h0k, &h0minusk);
    //imshow("h0k", h0k);
    //imshow("h0minusk", h0minusk);
    
    
    // spectrum texture(t)
    Mat hk(M, M, CV_64FC4);
    Mat heightmap4C(M, M, CV_64FC4);
    Mat heightmap1C(M, M, CV_64FC1);
    
    chrono::steady_clock::time_point start = Clock::now();
    chrono::steady_clock::time_point current;
    double t = 0;
    
    while (true) {
        // hk
        createHk(&h0k, &h0minusk, &hk, t);
        //imshow("hk", hk);
    
        // height texture (IFT)
        fft2D(&hk, &heightmap4C, INVERSE, SPECTRUM_IMG);
        //imshow("Height Map", heightmap4C);
        imshow4CTo1C("Height Map", &heightmap4C, &heightmap1C);
        //imshow4CTo1CNew("Height Map", &heightmap4C);
    
    
        // refresh window
        int key = waitKey(1);
        if (key == 'q') {
            break;
        }
    
        // advance time
        current = Clock::now();                                                    //to seconds
        t = (chrono::duration_cast<chrono::nanoseconds>(current - start).count() * 0.000000001) * SimulationSpeed;
        cout << t << "s" << endl;
    }
    
    

    /*
    // load Image
    Mat img = loadImage("./images/lena_color_256.tif");
    //imshow("Image", img);
    
    // 1Channel -> 4Channel
    Mat signal(M, M, CV_64FC4);
    im1CTo4C(&img, &signal);
    //imshow("signal4C", signal);
    
    
    // FT
    Mat X(M, M, CV_64FC4);
    fft2D(&signal, &X, FORWARD, NORMAL_IMG);
    //X *= 255;
    //imshow("X", X);
    
    // Shift
    Mat X_shift(M, M, CV_64FC4);
    shiftSpectrum(&X, &X_shift);
    X_shift *= 255;
    imshow("X Shift", X_shift);
    
    
    // IFT
    //X /= 255;
    Mat x(M, M, CV_64FC4);
    fft2D(&X, &x, INVERSE, NORMAL_IMG);
    imshow4CTo1CNew("x", &x);
    */

    // refresh window
    waitKey(0);

    return 0;
}



// General
// normalises output signal by essentially dividing it by the size of the signal
void normalise(Mat* pData, bool forward, bool spectrum) {
    if (spectrum) {
        if (forward) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    double* pixel = pData->ptr<double>(y, x);

                    pixel[0] = 0;
                    pixel[3] = 1;
                }
            }
        }
        else {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    double perm = 1.0;
                    int idx = int(fmod(int(y + x), 2));
                    perm = idx ? 1.0 : -1.0;

                    double* pixel = pData->ptr<double>(y, x);
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
                    double* pixel = pData->ptr<double>(y, x);

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
                    double* pixel = pData->ptr<double>(y, x);

                    pixel[0] = 0;
                    pixel[3] = 1;
                }
            }
        }
    }
}

// shifts spectrum from image corners to center and vice-versa
void shiftSpectrum(Mat* pX, Mat* pXShift) {
    for (int v = 0; v < M/2; v++) {
        memcpy(pXShift->ptr<double>(v + M/2, M/2), pX->ptr<double>(v), M*2 * sizeof(double));
        memcpy(pXShift->ptr<double>(v), pX->ptr<double>(v + M/2, M/2), M*2 * sizeof(double));

        memcpy(pXShift->ptr<double>(v + M/2, 0), pX->ptr<double>(v, M/2), M*2 * sizeof(double));
        memcpy(pXShift->ptr<double>(v, M/2), pX->ptr<double>(v + M/2, 0), M*2 * sizeof(double));
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

static void createButterflyTexture(Mat* pButterfly, bool forward) {
    #pragma omp parallel for collapse(2)
    for (int m = 0; m < M; m++) {
        for (int stage = 0; stage < BitsConst; stage++) {
            double k = fmod(double(m) * (double(M)/pow(2, stage + 1)), M);
            double exp = 2.0 * M_PI * k / double(M) * (forward ? -1 : 1);
            complex<double> twiddle(cos(exp), sin(exp));

            int butterflyspan = int(pow(2, stage));

            int butterflywing;
            if(fmod(m, pow(2, stage + 1)) < pow(2, stage))
                butterflywing = 1;
            else butterflywing = 0;


            double* pixel = pButterfly->ptr<double>(m, stage);
            if (stage == 0) {
                if (butterflywing == 1){
                             //M-1 - y
                    pixel[2] = twiddle.real();//r
                    pixel[1] = twiddle.imag();//g
                    pixel[0] = reverseBits(m, BitsConst);//b
                    pixel[3] = reverseBits(m + 1, BitsConst);//a
                }

                else{
                    pixel[2] = twiddle.real();//r
                    pixel[1] = twiddle.imag();//g
                    pixel[0] = reverseBits(m - 1, BitsConst);//b
                    pixel[3] = reverseBits(m, BitsConst);//a
                }
            }
            else {
                if (butterflywing == 1){
                    pixel[2] = twiddle.real();//r
                    pixel[1] = twiddle.imag();//g
                    pixel[0] = m;//b
                    pixel[3] = m + butterflyspan;//a
                }

                else{
                    pixel[2] = twiddle.real();//r
                    pixel[1] = twiddle.imag();//g
                    pixel[0] = m - butterflyspan;//b
                    pixel[3] = m;//a
                }
            }
        }
    }
}



// optimised 2D fast fourier transform that uses a texture of pre-computed twiddle factors and input indices
void fft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    if (forward) {
        fft2DWrapped(pSignal, pX, forward, spectrum, &ButterflyForward);
    }
    else {
        fft2DWrapped(pSignal, pX, forward, spectrum, &ButterflyInverse);
    }
}

//N.B. signal will be changed, should not be used again after running the FFT
void fft2DWrapped(Mat* pSignal, Mat* pX, bool forward, bool spectrum, Mat* pButterfly) {
    bool pingpong = 0;
    
    //horizontal butterflies
    for (int stage = 0; stage < BitsConst; stage++){
        if (pingpong == 0) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    horizontalButterfly(stage, y, x, pSignal, &Pingpong, pButterfly);
                }
            }
        }
        else if (pingpong == 1) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    horizontalButterfly(stage, y, x, &Pingpong, pSignal, pButterfly);
                }
            }
        }

        pingpong = !pingpong;
    }

    //vertical butterflies
    for (int stage = 0; stage < BitsConst; stage++){
        if (pingpong == 0) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    verticalButterfly(stage, y, x, pSignal, &Pingpong, pButterfly);
                }
            }
        }
        else if (pingpong == 1) {
            #pragma omp parallel for collapse(2)
            for (int y = 0; y < M; y++){
                for (int x = 0; x < M; x++){
                    verticalButterfly(stage, y, x, &Pingpong, pSignal, pButterfly);
                }
            }
        }

        pingpong = !pingpong;
    }


    //copy
    if (pingpong == 0) {
        memcpy(pX->data, pSignal->data, M*M*4 * sizeof(double));
    }
    else {
        memcpy(pX->data, Pingpong.data, M*M*4 * sizeof(double));
    }


    //Normalisation
    normalise(pX, forward, spectrum);
}

void horizontalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly) {
    double data[4];
    memcpy(&data, pButterfly->ptr<double>(x, stage), 4*sizeof(double));

    double p_[2];              
    int pIdx = (int)data[0]; //blue
    double* pixelInP = pPingpongIn->ptr<double>(y, pIdx);
    p_[0] = pixelInP[2]; //red
    p_[1] = pixelInP[1]; //green

    double q_[2];              
    int qIdx = (int)data[3]; //alpha
    double* pixelInQ = pPingpongIn->ptr<double>(y, qIdx);
    q_[0] = pixelInQ[2]; //red 
    q_[1] = pixelInQ[1]; //green


    complex<double> p(p_[0], p_[1]);
    complex<double> q(q_[0], q_[1]);
    complex<double> w(data[2], data[1]);

    complex<double> H = w*q + p;

    double* pixelOut = pPingpongOut->ptr<double>(y, x);
    pixelOut[2] = H.real();
    pixelOut[1] = H.imag();
}

void verticalButterfly(int stage, int y, int x, Mat* pPingpongIn, Mat* pPingpongOut, Mat* pButterfly) {
    double data[4];
    memcpy(&data, pButterfly->ptr<double>(y, stage), 4*sizeof(double));

    double p_[2];  
    int pIdx = (int)data[0]; //blue
    double* pixelInP = pPingpongIn->ptr<double>(pIdx, x);
    p_[0] = pixelInP[2]; //red
    p_[1] = pixelInP[1]; //green

    double q_[2];
    int qIdx = (int)data[3]; //alpha
    double* pixelInQ = pPingpongIn->ptr<double>(qIdx, x);
    q_[0] = pixelInQ[2]; //red 
    q_[1] = pixelInQ[1]; //green


    complex<double> p(p_[0], p_[1]);
    complex<double> q(q_[0], q_[1]);
    complex<double> w(data[2], data[1]);

    complex<double> H = w*q + p;

    double* pixelOut = pPingpongOut->ptr<double>(y, x);
    pixelOut[2] = H.real();
    pixelOut[1] = H.imag();
}


// My 2D fast fourier transform implementation that involved converting the recursive definition into an interative one
void fft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    //horizontal butterflies
    for (int n = 2; n <= M; n*=2) {
        #pragma omp parallel for
        for (int y = 0; y < M; y++) {
            horizontalButterfly_(n, y, pSignal, &Pingpong, forward);
        }
    }

    //vertical butterflies
    for (int n = 2; n <= M; n *= 2) {
        #pragma omp parallel for
        for (int x = 0; x < M; x++) {
            verticalButterfly_(n, x, &Pingpong, pX, forward);
        }
    }



    //Normalization
    normalise(pX, forward, spectrum);
}

void horizontalButterfly_(int n, int row, Mat* pPingpong0, Mat* pPingpong1, bool forward) {
    #pragma omp parallel for
    for (int v = 0; v < M; v+=n){

        double exp = 2 * M_PI / n * (forward ? -1 : 1);
        complex<double> w(1), wn(cos(exp), sin(exp));

        #pragma omp parallel for
        for (int i = 0; i < n/2; i++) {
            int evenIdx = v+  2*i;
            int oddIdx =  v+  2*i + 1;
            if (n==2) {
                evenIdx = reverseBits(evenIdx, BitsConst);
                oddIdx = reverseBits(oddIdx, BitsConst);
            }


            double* pixelP0Even = pPingpong0->ptr<double>(row, evenIdx);
            double* pixelP0Odd = pPingpong0->ptr<double>(row, oddIdx);
            complex<double> even(pixelP0Even[2], pixelP0Even[1]);
            complex<double> odd(pixelP0Odd[2], pixelP0Odd[1]);

            complex<double> butterfly_1 = even + w * odd;
            complex<double> butterfly_2 = even - w * odd;
            

            double* pixelP1First =  pPingpong1->ptr<double>(row, v+  i);
            double* pixelP1Second = pPingpong1->ptr<double>(row, v+  i + n/2);
            pixelP1First[2] = butterfly_1.real();   pixelP1First[1] = butterfly_1.imag();
            pixelP1Second[2] = butterfly_2.real();  pixelP1Second[1] = butterfly_2.imag();
            w *= wn;
        }
    }

    //even-odd merge step
    if (n < M){
        #pragma omp parallel for collapse(2)
        for (int v = 0; v < M; v+=2*n){
            for(int i = 0; i < 2*n; i+=2){
                memcpy(pPingpong0->ptr<double>(row, v + i),   pPingpong1->ptr<double>(row, v + i/2),     4*sizeof(double));
                memcpy(pPingpong0->ptr<double>(row, v + i+1), pPingpong1->ptr<double>(row, v + i/2 + n), 4*sizeof(double));
            }
        }
    }
}

void verticalButterfly_(int n, int col, Mat* pPingpong0, Mat* pPingpong1, bool forward) {
    #pragma omp parallel for
    for (int v = 0; v < M; v+=n){

        double exp = 2 * M_PI / n * (forward ? -1 : 1);
        complex<double> w(1), wn(cos(exp), sin(exp));

        #pragma omp parallel for
        for (int i = 0; i < n/2; i++) {
            int evenIdx = v+  2*i;
            int oddIdx =  v+  2*i + 1;
            if (n==2) {
                evenIdx = reverseBits(evenIdx, BitsConst);
                oddIdx = reverseBits(oddIdx, BitsConst);
            }


            double* pixelP0Even = pPingpong0->ptr<double>(evenIdx, col);
            double* pixelP0Odd = pPingpong0->ptr<double>(oddIdx, col);
            complex<double> even(pixelP0Even[2], pixelP0Even[1]);
            complex<double> odd(pixelP0Odd[2], pixelP0Odd[1]);

            complex<double> butterfly_1 = even + w * odd;
            complex<double> butterfly_2 = even - w * odd;

            
            double* pixelP1First = pPingpong1->ptr<double>(v+  i, col);
            double* pixelP1Second = pPingpong1->ptr<double>(v+  i + n/2, col);
            pixelP1First[2] = butterfly_1.real();   pixelP1First[1] = butterfly_1.imag();
            pixelP1Second[2] = butterfly_2.real();  pixelP1Second[1] = butterfly_2.imag();
            w *= wn;
        }
    }

    // even-odd merge step
    if (n < M){
        #pragma omp parallel for collapse(2)
        for (int v = 0; v < M; v+=2*n){
            for(int i = 0; i < 2*n; i+=2){
                memcpy(pPingpong0->ptr<double>(v + i, col),   pPingpong1->ptr<double>(v + i/2, col),     4*sizeof(double));
                memcpy(pPingpong0->ptr<double>(v + i+1, col), pPingpong1->ptr<double>(v + i/2 + n, col), 4*sizeof(double));
            }
        }
    }
}


// 2D fourier transform broken down as a series of 1D FTs on the rows followed by 1D FTs on the resultant columns
void ft2D(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    // Row DFTS
    #pragma omp parallel for
    for (int y = 0; y < M; y++) {
        ft1D(pSignal, &Pingpong, forward, true, y);
    }

    // Column DFTS
    #pragma omp parallel for
    for (int x = 0; x < M; x++) {
        ft1D(&Pingpong, pX, forward, false, x);
    }


    // Normalization
    normalise(pX, forward, spectrum);
}

void ft1D(Mat* pSignal, Mat* pX, bool forward, bool horizontal, int row_col) {
    double factor = forward ? -1 : 1;

    if (horizontal) {
        #pragma omp parallel for
        for (int k = 0; k < M; k++) {

            double sum_real = 0;
            double sum_imag = 0;

            //#pragma omp parallel for reduction(+:sum_real) reduction(+:sum_imag)   //BROKEN
            for (int n = 0; n < M; n++) {
                double real =          cos(((2 * M_PI) / M) * k * n);
                double imag = factor * sin(((2 * M_PI) / M) * k * n);

                double* pixel = pSignal->ptr<double>(row_col, n);
                complex<double> val = complex<double>(pixel[2], pixel[1]) * complex<double>(real, imag);
                sum_real += val.real();
                sum_imag += val.imag();
            }

            double* pixel = pX->ptr<double>(row_col, k);
            pixel[2] = sum_real;
            pixel[1] = sum_imag;
        }
    }
    else {
        #pragma omp parallel for
        for (int k = 0; k < M; k++) {

            double sum_real = 0;
            double sum_imag = 0;

            //#pragma omp parallel for reduction(+:sum_real) reduction(+:sum_imag)  //BROKEN
            for (int n = 0; n < M; n++) {
                double real =          cos(((2 * M_PI) / M) * k * n);
                double imag = factor * sin(((2 * M_PI) / M) * k * n);

                double* pixel = pSignal->ptr<double>(n, row_col);
                complex<double> val = complex<double>(pixel[2], pixel[1]) * complex<double>(real, imag);
                sum_real += val.real();
                sum_imag += val.imag();
            }

            double* pixel = pX->ptr<double>(k, row_col);
            pixel[2] = sum_real;
            pixel[1] = sum_imag;
        }
    }
}


// default 2D fourier transform implemented directly from textbook definition
void ft2D_(Mat* pSignal, Mat* pX, bool forward, bool spectrum) {
    double factor = (forward ? -1.0 : 1.0);
    #pragma omp parallel for collapse(2)
    for (int v = 0; v < M; v++) {
        for (int u = 0; u < M; u++) {

            double sum_real = 0;
            double sum_imag = 0;

            #pragma omp parallel for collapse(2) reduction(+:sum_real) reduction(+:sum_imag)
            for (int y = 0; y < M; y++) {
                for (int x = 0; x < M; x++) {
                    double exp = (2 * M_PI) * (x*u + y*v) / M;
                    double real = cos(exp);
                    double imag = factor * sin(exp);

                    double* pixel = pSignal->ptr<double>(y, x);
                    complex<double> val = complex<double>(pixel[2], pixel[1]) * complex<double>(real, imag);
                    sum_real += val.real();
                    sum_imag += val.imag();
                }
            }

            //different channels
            double* pixel = pX->ptr<double>(v, u);
            pixel[2] = sum_real;
            pixel[1] = sum_imag;
        }
    }

    normalise(pX, forward, spectrum);
}



// Create Oceanographic Spectrum
// rotate spectrum across the complex plane to move simulation through time
void createHk(Mat* pH0k, Mat* pH0minusk, Mat* pHk, double t) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < M; y++){
        for (int x = 0; x < M; x++) {

            double K1 = 2 * M_PI * (x - M/2) / L;
            double K2 = 2 * M_PI * (y - M/2) / L;
            double k = sqrt(K1*K1 + K2*K2);

            double omega = sqrt(g * k);
            double expo = omega * t;
            complex<double> comp_1(cos(expo), sin(expo));
            complex<double> comp_2(comp_1.real(), -comp_1.imag());

            double* pixelH0k = pH0k->ptr<double>(y, x);
            complex<double> h0k_val(pixelH0k[2], pixelH0k[1]);

            double* pixelH0minusk = pH0minusk->ptr<double>(y, x);
            complex<double> h0minusk_val(pixelH0minusk[2], pixelH0minusk[1]);

            complex<double> hk_val = h0k_val * comp_1  +  h0minusk_val * comp_2;
            double* pixelHk = pHk->ptr<double>(y, x);
            pixelHk[2] = hk_val.real();
            pixelHk[1] = hk_val.imag();
            pixelHk[0] = 0;
            pixelHk[3] = 1;
        }
    }
}

// create initial spectrum
void createH0(Mat* pNoise, Mat* pH0k, Mat* pH0minusk) {
    const double bound = 4000;

    #pragma omp parallel for collapse(2)
    for (int y = 0; y < M; y++){
        for (int x = 0; x < M; x++) {

            double K1 = 2 * M_PI * (x - M/2) / L;
            double K2 = 2 * M_PI * (y - M/2) / L;
            double k = sqrt(K1*K1 + K2*K2);
            
            double suppressionFactor = exp(-k*k*l*l);
            

            double* pixelNoise = pNoise->ptr<double>(y, x);

            double h0k_val = max(-bound, min(bound, sqrt((suppressionFactor * phillips(K1, K2, k)) * 0.5)));
            double* pixelH0k = pH0k->ptr<double>(y, x);
            pixelH0k[2] = pixelNoise[2] * h0k_val;
            pixelH0k[1] = pixelNoise[1] * h0k_val;

            double h0minusk_val = max(-bound, min(bound, sqrt((suppressionFactor * phillips(-K1, -K2, k)) * 0.5)));
            double* pixelH0minusk = pH0minusk->ptr<double>(y, x);
            pixelH0minusk[2] = pixelNoise[0] * h0minusk_val;
            pixelH0minusk[1] = pixelNoise[3] * h0minusk_val;
        }
    }
}

// Phillips Spectrum equation
double phillips(double K1, double K2, double k) {
    double K1_N = K1/k;
    double K2_N = K2/k;
    double W1_N = W1/w;
    double W2_N = W2/w;

    return A * (exp(-pow(k*L_, -2)) * pow(k, -4)) * pow(K1_N*W1_N + K2_N*W2_N, 2);
}

// generate a texture of pseudorandom noise
void createNoiseTexture(Mat* pNoise) {
    double lower_bound = -1;
    double upper_bound = 1;
    uniform_real_distribution<double> unif(lower_bound, upper_bound);
    default_random_engine re;


    for (int y = 0; y < M; y++){
        for (int x = 0; x < M; x++) {
            double* pixel = pNoise->ptr<double>(y, x);

            pixel[2] = unif(re);
            pixel[1] = unif(re);
            pixel[0] = unif(re);
            pixel[3] = unif(re);
        }
    }
}



// Load image from disk
Mat loadImage(string image_path) {
    Mat img = imread(image_path, IMREAD_GRAYSCALE);
    img.convertTo(img, CV_64FC1);
    img /= 255;
    resize(img, img, { M, M }, 0, 0, cv::INTER_NEAREST);

    return img;
}



// Convert/display images
// convert between grayscale and BGRA
void im1CTo4C(Mat* pImage1C, Mat* pImage4C) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            double* pixel1C = pImage1C->ptr<double>(y, x);
            double* pixel4C = pImage4C->ptr<double>(y, x);
            pixel4C[2] = *pixel1C;
            pixel4C[1] = *pixel1C;
            pixel4C[0] = *pixel1C;
            pixel4C[3] = 1;
        }
    }
}

void im4CTo1C(Mat* pImage4C, Mat* pImage1C) {
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            double* pixel4C = pImage4C->ptr<double>(y, x);
            double* pixel1C = pImage1C->ptr<double>(y, x);
            *pixel1C = pixel4C[2];
        }
    }
}


// convert and display
void imshow1CTo4C(string name, Mat* pImage1C, Mat* pImage4C) {
    im1CTo4C(pImage1C, pImage4C);
    imshow(name, *pImage4C);
}

void imshow4CTo1C(string name, Mat* pImage4C, Mat* pImage1C){
    im4CTo1C(pImage4C, pImage1C);
    imshow(name, *pImage1C);
}

void imshow1CTo4CNew(string name, Mat* pImage1C) {
    Mat image_4C(M, M, CV_64FC4);
    im1CTo4C(pImage1C, &image_4C);

    imshow(name, image_4C);
}

void imshow4CTo1CNew(string name, Mat* pImage4C){
    Mat image_1C(M, M, CV_64FC1);
    im4CTo1C(pImage4C, &image_1C);

    imshow(name, image_1C);
}