#ifndef XNVUTIL_H
#define XNVUTIL_H

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

#endif // XNVUTIL_H
