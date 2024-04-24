#pragma once

#include <vector>
#include <thread>

#ifdef IPP_IMPL

#include "ipp.h"
// Suppressed clangd warning by doing all transitive includes
#include "ipp_ext.h" // IWYU pragma: export

template <typename T>
using Vector = ippe::vector<T>;

#else

template <typename T>
using Vector = std::vector<T>;

#endif

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

    // main runtime methods
    void xcorr_array(
        const T* src,
        const int srclen,
        const int startIdx, 
        const int endIdx, 
        const int idxStep,
        U *productpeaks,
        int *freqlistinds,
        int outputlength
    );

    // Used for resizing outputs correctly
    // or checking if output lengths are correct
    int getOutputLength(int startIdx, int endIdx, int idxStep)
    {
        // Accounts for the start, end and step
        int length = (endIdx - startIdx) / idxStep;
        // May allow for 1 more depending on the step modulo
        if ((endIdx - startIdx) % idxStep != 0)
            length += 1;
        return length;
    }

private:
    int m_cutoutlen;
    int m_num_threads = 1;

    Vector<T> m_cutout;
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
        U *productpeaks,
        int *freqlistinds
    );

};

/*
===============================================
IPP Implementation

As far as possible, this uses ipp_ext templates,
to reduce the need for specialization.
===============================================
*/
#ifdef IPP_IMPL

// Constructor
template <typename T, typename U>
XcorrFFT<T, U>::XcorrFFT(
    const T* cutout, int cutoutlen, int num_threads, bool autoConj
) : m_cutoutlen{cutoutlen}, m_num_threads{num_threads}
{
    // Store cutout internally
    m_cutout.resize(m_cutoutlen);
    ippe::Copy(cutout, m_cutout.data(), m_cutoutlen);
    if (autoConj)
        ippe::convert::Conj_I(m_cutout.data(), m_cutoutlen);

    // compute the norm squared and store it
    Ipp64f norm2;
    ippe::math::Norm_L2(m_cutout.data(), m_cutoutlen, &norm2);
    m_cutoutNormSq = static_cast<U>(norm2*norm2);
}

// Main private method for computation
template <typename T, typename U>
inline void XcorrFFT<T, U>::xcorr_thread(
    const T* src,
    const int srclen,
    const int startIdx,
    const int endIdx,
    const int idxStep,
    const int tIdx,
    U *productpeaks,
    int *freqlistinds
){
    int outputlen = getOutputLength(startIdx, endIdx, idxStep);

    // Output values for productpeaks/freqlistinds
    U maxval;
    int maxind;
    // This norm is always 64f
    Ipp64f slicenorm;

    // local fft object
    ippe::DFTCToC<T> fftobj((size_t)m_cutoutlen);
    // and local workspace
    ippe::vector<T> work_fc_1(m_cutoutlen);
    ippe::vector<T> work_fc_2(m_cutoutlen);

    int i;
    for (int t = tIdx; t < outputlen; t += m_num_threads)
    {
        // Define the accessor index for the src
        i = startIdx + t * idxStep; 

        // printf("In thread %d, startIdx %d, on output %d/%d\n", tIdx, i, t, outputlen);

        // Don't compute if we're out of range
        if (i < 0 || i + m_cutoutlen > srclen)
        {
            productpeaks[t] = 0.0f;
            freqlistinds[t] = 0;
            continue;
        }

        // First we multiply, use the first workspace
        // printf("Attempting Mul\n");
        try{
            ippe::math::Mul(
                m_cutout.data(),
                &src[i],
                work_fc_1.data(),
                m_cutoutlen
            );
        }
        catch (std::runtime_error& e)
        {
            printf("Mul failed %s\n", e.what());
        }
        // printf("Completed Mul\n");

        // Then we fft the output, use the second workspace
        try{
            // m_ffts[tIdx].fwd(
            fftobj.fwd(
                work_fc_1.data(),
                work_fc_2.data()
            );
        }
        catch(const std::exception& e){
            printf("Exception for thread %d: %s\n", tIdx, e.what());
            // printf("workspace size: %zd\n", m_work_fc_1[tIdx].size());
            // printf("workspace size: %zd\n", m_work_fc_2[tIdx].size());
        }
        // printf("Completed FFT\n");

        // Get abs squared, reuse first workspace
        ippe::convert::PowerSpectr(
            work_fc_2.data(),
            reinterpret_cast<U*>(work_fc_1.data()), // note that this uses the 'first half' of the alloc'ed memory
            m_cutoutlen
        );

        // Get the max index, and the associated value
        ippe::stats::MaxIndx(
            reinterpret_cast<U*>(work_fc_1.data()),
            m_cutoutlen, 
            &maxval, &maxind
        );

        // get the norm sq for this slice
        ippe::stats::Norm_L2(
            &src[i], m_cutoutlen, &slicenorm
        );

        // compute the output with scaling
        productpeaks[t] = maxval / m_cutoutNormSq / static_cast<U>(slicenorm * slicenorm);
        freqlistinds[t] = maxind;
    }
}

// Public method with C-like array
template <typename T, typename U>
void XcorrFFT<T, U>::xcorr_array(
    const T* src,
    const int srclen,
    const int startIdx, 
    const int endIdx, 
    const int idxStep,
    U *productpeaks,
    int *freqlistinds,
    int outputlength
){
    // check the output length is 'correct' as a validation mechanic
    if (getOutputLength(startIdx, endIdx, idxStep) != outputlength)
    {
        throw std::runtime_error("Output length is not correct");
    }

    // start threads to iterate over
    m_threads.resize(m_num_threads);
    for (int i = 0; i < m_num_threads; i++)
    {
        // printf("Launching thread %d\n", i);
        m_threads[i] = std::thread(
            &XcorrFFT::xcorr_thread,
            this, 
            src, 
            srclen, 
            startIdx, 
            endIdx, 
            idxStep,
            i, // thread id
            productpeaks,
            freqlistinds
        );
    }

    // wait for all threads to finish
    for (int i = 0; i < m_num_threads; i++)
    {
        m_threads[i].join();
    }
}


#endif

