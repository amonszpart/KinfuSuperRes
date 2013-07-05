#ifndef MY_PCD_VIEWER_H
#define MY_PCD_VIEWER_H

#include <string>

namespace am
{
    class MyPCDViewer
    {
        public:
            MyPCDViewer();

            int
            run( const std::string &file_name );
    };

} // ns am

#endif // MY_PCD_VIEWER_H
