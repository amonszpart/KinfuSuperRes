#ifndef MACOMMON_H
#define MACOMMON_H

#define SAFE_DELETE(a) if(a)delete a;
#define SAFE_DELETE_ARR(a) if(a)delete [] a;

#define SAFE_DELETE_ARR2(a,l) { for ( unsigned i = 0U; i < l; ++i ) { SAFE_DELETE( a[i] ); } delete [] a; }

#define AM_SUCCESS 0;

typedef std::map<std::string,cv::Mat> TMatDict;

#define CHECK_RC(rc, what)											\
    if (rc != XN_STATUS_OK)											\
    {																\
        printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
        return rc;													\
    }

#define CHECK_RC_ERR(rc, what, error)			\
{												\
    if (rc == XN_STATUS_NO_NODE_PRESENT)		\
    {											\
        XnChar strError[1024];					\
        errors.ToString(strError, 1024);		\
        printf("%s\n", strError);				\
    }											\
    CHECK_RC(rc, what)							\
}


#define CHECK_ERRORS(rc, errors, what)		\
    if (rc == XN_STATUS_NO_NODE_PRESENT)	\
    {										\
        XnChar strError[1024];				\
        errors.ToString(strError, 1024);	\
        printf("%s\n", strError);			\
        return (rc);						\
    }

#endif // MACOMMON_H
