#include "XcorrFFT.h"
#include <iostream>

int main()
{
    // create some data
    ippe::vector<Ipp32fc> data(100);
    for (int i = 0; i < data.size(); i++){
        data[i].re = i; data[i].im = i+1.0f;
    }

    int cutoutlen = 30;
    ippe::vector<Ipp32fc> cutout(cutoutlen);
    // copy from a place in data
    int c = 20;
    for (int i = 0; i < cutoutlen; i++){
        cutout[i].re = data[c+i].re;
        cutout[i].im = data[c+i].im;
    }


    // create xcorr object
    XcorrFFT<Ipp32fc, Ipp32f> xcfft(cutout.data(), cutout.size(), 1, true);

    // define xcorr limits
    int startIdx = 2;
    int endIdx = data.size();
    int idxStep = 3;
    ippe::vector<Ipp32f> productpeaks(xcfft.getOutputLength(startIdx, endIdx, idxStep));
    ippe::vector<Ipp32s> freqlistinds(productpeaks.size());
    printf("Output size = %zd\n", productpeaks.size());

    // loop arbitrarily many times to see the error
    for (int i = 0; i < 2; i++)
    {
        printf("Performing xcorr...\n");
        try{
            xcfft.xcorr_array(
                data.data(), data.size(), 
                startIdx, endIdx, idxStep,
                productpeaks.data(),
                reinterpret_cast<int*>(freqlistinds.data()),
                static_cast<int>(productpeaks.size())
            ); // overshoot, but it should write 0s
        }
        catch(std::exception &e)
        {
            std::cout << e.what() << std::endl;
        }
    }
    printf("Xcorr complete\n");

    for (int i = 0; i < productpeaks.size(); i++){
        printf("Peak [%d]: %f, fidx %d \n", startIdx + idxStep * i, productpeaks.at(i), freqlistinds.at(i));
    }

    printf("Complete\n");

    return 0;
}
