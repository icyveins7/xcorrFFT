#pragma once

#include "ipp.h"
#include "ipp_ext.h"
#include <vector>
#include <thread>

// T: complex type, U: real type
// This is so it's more generic and can be used with
// std::complex-like types that encapsulate,
// as well as IPP-like types that strictly define the complex type.
template <typename T, typename U>
class XcorrFFT
{
public:
    XcorrFFT(const T* cutout, int cutoutlen, int num_threads, bool autoConj);
    ~XcorrFFT() = default;

    // main runtime method
    void xcorr(
        const T* src,
        const int srclen,
        const int startIdx, 
        const int endIdx, 
        const int idxStep
    );

    void xcorr_array(
        const T* src,
        const int srclen,
        const int startIdx, 
        const int endIdx, 
        const int idxStep,
        float *productpeaks,
        int *freqlistinds,
        int outputlength
    );

    // output vectors
    std::vector<U> m_productpeaks;
    std::vector<int> m_freqlistinds;

private:
    int m_cutoutlen;
    int m_num_threads = 1;

    ippe::vector<T> m_cutout;
    U m_cutoutNormSq;

    // threads
    std::vector<std::thread> m_threads;

    // internal work method
    void xcorr_thread(
        const T* src,
        const int srclen,
        const int startIdx, 
        const int endIdx, 
        const int idxStep,
        const int tIdx,
        float *productpeaks,
        int *freqlistinds
    );

    int getOutputLength(int startIdx, int endIdx, int idxStep);


};
