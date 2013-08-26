#version 150

in vec4 position_in_world_space;
//in vec2 out_texCoord;
in vec3 out_normal;

out vec4  outputF;
out uvec3 ids;
//flat vec3 frag_eyePos;
uniform vec3 eyePos;

void main()
{
    vec4 pos_cartesian = vec4( position_in_world_space.xyz / position_in_world_space.w, 1.0 );

    float dist = clamp(
                    distance(pos_cartesian, vec4( eyePos.x, eyePos.y, eyePos.z,1.0)) / 10.01f,
                    0.f,
                    1.f );
    //outputF = vec4( out_normal.x/10.f, dist, gl_PrimitiveID + 1, 1.0 );
    //outputF = vec4( out_normal.x/10.f, dist, out_normal.x/530679.f, 1.0 );
    outputF = vec4( dist, dist, dist, 1.0 );
    ids = uvec3( 1, 2, uint(out_normal.x) );
}


/*uniform sampler2D qt_Texture0;
varying vec4 qt_TexCoord0;

void main(void)
{
    gl_FragColor = texture2D(qt_Texture0, qt_TexCoord0.st);
}*/

