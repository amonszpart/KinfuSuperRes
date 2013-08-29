#version 150

uniform vec3 eyePos;

in vec4 position_in_world_space;
in vec3 frag_vxid_fid;
flat in uint flat_fid;

out vec4  outputF;
out uvec3 ids;
out vec4  dists;

void main()
{
    vec3 pos_cartesian = vec3( position_in_world_space.xyz / position_in_world_space.w );

    float dist = distance( pos_cartesian, eyePos );
    outputF = vec4 ( dist, float(frag_vxid_fid.x), 0.55f, 1.0 );
    ids     = uvec3( /*    vertex id: */ round(frag_vxid_fid.x),
                     /*      face id: */ round(frag_vxid_fid.y),
                     /* flat face id: */ flat_fid
                   );
    dists = vec4( dist/10.1f, 3.f, dist/10.1f, dist/10.1f );
}

//ids     = uvec3( uint(round(out_normal.x)), uint(round(out_normal.y)), uint(round(out_normal.x)) );
