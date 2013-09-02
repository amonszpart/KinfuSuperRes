#version 150

// camera position
uniform vec3 eyePos;

// interpolated vertex position
in vec4 position_in_world_space;
// closes vertex ids, face ids
in vec3 frag_vxid_fid;
// face id uninterpolated
flat in uint flat_fid;

// color_attachment0
out vec4  outputF;
//float outputF;
// color_attachment1
out uvec3 ids;
// color_attachment2 - unused
out vec4 dists;

void main()
{
    vec3 pos_cartesian = vec3( position_in_world_space.xyz / position_in_world_space.w );

    // w ==1, so it doesn't matter
    //vec3 pos_cartesian = vec3( position_in_world_space.xyz );

    // calculate distance from camera
    float dist = distance( pos_cartesian, eyePos );

    // x: scale distance to 0mm...10001millimeters, from 0..10.1f in the PLY's space
    // y,z: not used
    //outputF = vec4 ( dist * 1000.f, dist/10.1f, dist/10.1f, 1.0 );
    //outputF = dist * 1000.f;
    //outputF = vec4( dist * 1000.f, 0, 0, 1 );
    outputF = vec4( dist * 1000.f, 0, 0, 0 );

    // debug check for eye position (is ok)
    //outputF = vec4 ( eyePos.x, eyePos.y, eyePos.z, 1.0 );

    // output face ids to color_attachment1
    ids     = uvec3( /*    vertex id: */ round(frag_vxid_fid.x),
                     /*      face id: */ round(frag_vxid_fid.y),
                     /* flat face id: */ flat_fid
                   );

    // output distances to color_attachment2 - not used
    //dists = vec4( dist/10.1f, 3.f, dist/10.1f, dist/10.1f );
}
