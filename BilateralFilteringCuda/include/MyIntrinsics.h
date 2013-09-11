#ifndef MYINTRINSICS_H
#define MYINTRINSICS_H

#include "calibration_cuda_constants_prism4.h"

struct MyIntrinsics
{
        float fx, fy, cx, cy;
        float k1, k2, p1, p2, k3, alpha;

        MyIntrinsics()
            : fx(0.f), fy(0.f), cx(0.f), cy(0.f), k1(0.f), k2(0.f), p1(0.f), p2(0.f), k3(0.f), alpha(0.f)
        {}

        MyIntrinsics( float fx, float fy, float cx, float cy,
                      float k1, float k2, float p1, float p2, float k3,
                      float alpha )
            : fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), p1(p1), p2(p2), k3(k3), alpha(alpha)
        {}

        MyIntrinsics( INTRINSICS_CAMERA_ID cid, bool use_distort = true )
        {
            if ( cid == DEP_CAMERA )
            {
                fx = FX_D;
                fy = FY_D;
                cx = CX_D;
                cy = CY_D;

                if ( use_distort )
                {
                    k1 = K1_D;
                    k2 = K2_D;
                    p1 = P1_D;
                    p2 = P2_D;
                    k3 = K3_D;
                    alpha = ALPHA_D;
                }
            }
            else if ( cid == RGB_CAMERA )
            {
                fx = FX_RGB;
                fy = FY_RGB;
                cx = CX_RGB;
                cy = CY_RGB;

                if ( use_distort )
                {
                    k1 = K1_RGB;
                    k2 = K2_RGB;
                    p1 = P1_RGB;
                    p2 = P2_RGB;
                    k3 = K3_RGB;
                    alpha = ALPHA_RGB;
                }
            }
            else
            {
                //std::cerr << "MyIntrinsics: wrong camera ID" << std::endl;
            }
        }
};

#endif // MYINTRINSICS_H
