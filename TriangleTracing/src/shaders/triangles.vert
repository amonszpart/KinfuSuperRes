#version 150

uniform mat4 viewMatrix, projMatrix, modelMatrix;

in vec4 position;
in vec3 color;

out vec3 Color;
out vec4 position_in_world_space;
flat out vec3 Color2;

void main()
{
    Color = color;
    Color2 = color;
    position_in_world_space = viewMatrix * modelMatrix * position;
    gl_Position = projMatrix * position_in_world_space;

    //gl_Position = projMatrix * viewMatrix * modelMatrix * vec4(position,1.0);
}

/*attribute vec4 qt_Vertex;
attribute vec4 qt_MultiTexCoord0;
uniform mat4 qt_ModelViewProjectionMatrix;
varying vec4 qt_TexCoord0;

void main(void)
{
    gl_Position = qt_ModelViewProjectionMatrix * qt_Vertex;
    qt_TexCoord0 = qt_MultiTexCoord0;
}*/
