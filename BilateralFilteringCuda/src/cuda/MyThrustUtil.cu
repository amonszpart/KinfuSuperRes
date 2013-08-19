#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int testKernel()
{
    // initialize all ten integers of a device_vector to 1
      thrust::device_vector<int> D(10, 1);

      // set the first seven elements of a vector to 9
      thrust::fill(D.begin(), D.begin() + 7, 9);

      // initialize a host_vector with the first five elements of D
      thrust::host_vector<int> H(D.begin(), D.begin() + 5);

      // set the elements of H to 0, 1, 2, 3, ...
      thrust::sequence(H.begin(), H.end());

      // copy all of H back to the beginning of D
      thrust::copy(H.begin(), H.end(), D.begin());

      // print D
      for(int i = 0; i < D.size(); i++)
      {
        std::cout << "D[" << i << "] = " << D[i] << std::endl;
      }

      return 0;
}

/////////// Squre Diff Scalar [trunc] ////////////////

struct sqr_diff_scalar_functor
{
  const float a;

  sqr_diff_scalar_functor(float _a) : a(_a) {}

  __host__ __device__
  float operator()(const float& x) const
  {
      return (x-a) * (x-a);
  }
};

struct sqr_diff_scalar_trunc_functor
{
  const float a;
  const float trunc;

  sqr_diff_scalar_trunc_functor(float _a, float _trunc)
      : a(_a), trunc(_trunc) {}

  __host__ __device__
  float operator()(const float& x) const
  {
      return min( (a-x) * (a-x), trunc );
  }
};

int squareDiffScalarKernel( float *a, int size, float x, float *out, float truncAt )
{
    thrust::device_vector<float> d_a  ( a, a + size );
    thrust::device_vector<float> d_out( size );

    if ( truncAt < 0.f )
    {
        thrust::transform( d_a.begin(), d_a.end(), d_out.begin(),
                           sqr_diff_scalar_functor(x) );
    }
    else
    {
        thrust::transform( d_a.begin(), d_a.end(), d_out.begin(),
                           sqr_diff_scalar_trunc_functor(x,truncAt) );
    }

    // copy to host
    thrust::host_vector<float> h_out( size );
    h_out = d_out;

    // copy to memspace
    std::copy( h_out.begin(), h_out.end(), out );

    return 0;
}

/////////// Squre Diff ////////////////

int squareDiffKernel( float *a, int size, float* b, float *out, float truncAt )
{
    std::cerr << "squareDiffKernel not impl yet" << std::endl;

    return 0;
}

/////////// minMaskedCopy ////////////////

/*if ( C(d) < minC(d_min) )
{
    // copy previous d's cost to (d-1)'s cost
    minCm1(d_min) = C( d-1 );
    // store current d
    d_min = d;
    // store current cost
    minC(d_min) = C(d);
}
else
{
    if ( minD == d-1 )
        minCp1(minD) = C(d);
}*/



struct min_masked_copy_functor
{
        enum {
            CPREV  = 0,
            C      = 1,
            MINC   = 2,
            MINDS  = 3,
            MINCM1 = 4,
            MINCP1 = 5
        };

        const float d;
        min_masked_copy_functor( float _d )
            : d(_d) {}

        // Tuple is <Cprev, C, minC, minDs, minCm1, minCp1 >
        template <typename Tuple>
        __host__ __device__
        void operator()( Tuple t )
        {
            if ( thrust::get<C>(t) < thrust::get<MINC>(t) )      // if ( C[i] < minC[i] )
            {
                // copy previous d's cost to (d-1)'s cost
                thrust::get<MINCM1>(t) = thrust::get<CPREV>(t); // minCm1[i] = Cprev[i]
                // store current d
                thrust::get<MINDS>(t) = d;                      // minDs[i] = d
                // store current C(d)
                thrust::get<MINC>(t) = thrust::get<C>(t);       // minC[i] = C[i]
            }
            else if ( thrust::get<MINDS>(t) == d-1 )            // if ( minDs[i] == d-1 )
            {
                thrust::get<MINCP1>(t) = thrust::get<C>(t);    // minCp1[i] = C[i]
            }
        }

};

/*
 * @brief       performs "minDs( find(C < minC) ) = d"
 * @param C     estimated cost for d depth
 * @param minC  minimum cost for previous d-s
 * @param d     current d depth
 * @param minDs previous d-s with minimum cost, serves as output
 */
int minMaskedCopyKernel(
        float*  const& Cprev ,
        float*  const& C     ,
        float          d     ,
        int            size  ,
        float*       & minC  ,
        float*       & minDs ,
        float*       & minCm1,
        float*       & minCp1 )
{
    // copy input to device
    thrust::device_vector<float> d_Cprev ( Cprev , Cprev  + size );
    thrust::device_vector<float> d_C     ( C     , C      + size );
    thrust::device_vector<float> d_minC  ( minC  , minC   + size );
    thrust::device_vector<float> d_minDs ( minDs , minDs  + size );
    thrust::device_vector<float> d_minCm1( minCm1, minCm1 + size );
    thrust::device_vector<float> d_minCp1( minCp1, minCp1 + size );


    // apply the transformation
    thrust::for_each(  thrust::make_zip_iterator(thrust::make_tuple(
                                                     d_Cprev .begin(),
                                                     d_C     .begin(),
                                                     d_minC  .begin(),
                                                     d_minDs .begin(),
                                                     d_minCm1.begin(),
                                                     d_minCp1.begin() )),
                       thrust::make_zip_iterator(thrust::make_tuple(
                                                     d_Cprev .end  (),
                                                     d_C     .end  (),
                                                     d_minC  .end  (),
                                                     d_minDs .end  (),
                                                     d_minCm1.end  (),
                                                     d_minCp1.end  () )),
                       min_masked_copy_functor(d)                          );

    // copy to host
    thrust::host_vector<float> h_tmp( size );

    h_tmp = d_minC;
    std::copy( h_tmp.begin(), h_tmp.end(), minC );
    h_tmp = d_minDs;
    std::copy( h_tmp.begin(), h_tmp.end(), minDs );
    h_tmp = d_minCm1;
    std::copy( h_tmp.begin(), h_tmp.end(), minCm1 );
    h_tmp = d_minCp1;
    std::copy( h_tmp.begin(), h_tmp.end(), minCp1 );

    return 0;
}

/////////// SubpixelRefine ////////////////

struct subpixel_refine_functor
{
        enum {
            MINC   = 0,
            MINCM1 = 1,
            MINCP1 = 2,
            MINDS  = 3
        };
        // Tuple is <minC, minCm1, minCp1, minDs>
        template <typename Tuple>
        __host__ __device__
        void operator()(Tuple t)
        {
            float a1 = thrust::get<MINCP1>(t) - thrust::get<MINCM1>(t);
            float a2 = ( 2.f * (thrust::get<MINCP1>(t) + thrust::get<MINCM1>(t) - 2.f * thrust::get<MINC>(t)) );
            float a3 = a1 / a2;
            float d = thrust::get<MINDS>(t);
            a3 = min( max( a3, d * -2.f), d * 2.f );
            thrust::get<MINDS>(t) -= a3;
        }
};

int subpixelRefineKernel( float*  const& minC  ,
                          float*  const& minCm1,
                          float*  const& minCp1,
                          int            size  ,
                          float*       & minDs  )
{
    // copy input to device
    thrust::device_vector<float> d_minC  ( minC  , minC   + size );
    thrust::device_vector<float> d_minCm1( minCm1, minCm1 + size );
    thrust::device_vector<float> d_minCp1( minCp1, minCp1 + size );
    thrust::device_vector<float> d_minDs ( minDs , minDs  + size );

    // apply the transformation
    thrust::for_each(  thrust::make_zip_iterator(thrust::make_tuple(
                                                     d_minC  .begin(),
                                                     d_minCm1.begin(),
                                                     d_minCp1.begin(),
                                                     d_minDs .begin() )),
                       thrust::make_zip_iterator(thrust::make_tuple(
                                                     d_minC  .end(),
                                                     d_minCm1.end(),
                                                     d_minCp1.end(),
                                                     d_minDs .end() )),
                       subpixel_refine_functor()                         );

    // copy to host
    thrust::host_vector<float> h_tmp( size );
    h_tmp = d_minDs;
    std::copy( h_tmp.begin(), h_tmp.end(), minDs );

    return 0;
}


