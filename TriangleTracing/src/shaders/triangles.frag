#version 150

in vec3 Color;

out vec4 outputF;
in vec4 position_in_world_space;
flat in vec3 Color2;

void main()
{
    vec4 tmp = vec4(position_in_world_space.xyz / position_in_world_space.w, 1.0);

    float dist = clamp( distance( tmp, vec4(0.0, 0.0, 0.0, 1.0) ) / 20.f, 0.f, 1.f );
    outputF = vec4( Color2.x/10.f, dist, gl_PrimitiveID + 1, 1.0 );
}


/*uniform sampler2D qt_Texture0;
varying vec4 qt_TexCoord0;

void main(void)
{
    gl_FragColor = texture2D(qt_Texture0, qt_TexCoord0.st);
}*/

