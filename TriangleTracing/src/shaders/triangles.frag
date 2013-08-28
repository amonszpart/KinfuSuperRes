#version 150

uniform vec3 eyePos;

in vec4 position_in_world_space;
in vec3 out_normal;

out vec4  outputF;
out uvec3 ids;

void main()
{
    vec3 pos_cartesian = vec3( position_in_world_space.xyz / position_in_world_space.w );

    float dist = distance( pos_cartesian, eyePos ) / 10.01f;
    outputF = vec4 ( dist, out_normal.x, dist, 1.0 );
    ids     = uvec3( uint(out_normal.x), uint(out_normal.y), uint(out_normal.x) );
}
