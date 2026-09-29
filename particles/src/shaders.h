// -----------------------------------------------------------------------------
//  shaders.h — the two GLSL programs the particle system needs
// -----------------------------------------------------------------------------
#pragma once

// -----------------------------------------------------------------------------
//  Points. Each particle is one GL_POINTS vertex. The vertex shader maps world
//  space (x in [-aspect, aspect], y in [-1, 1]) straight to clip space, and
//  turns the packed speed into a point size. The fragment shader cuts the
//  square point sprite into a soft disc.
//
//  The alpha falloff combined with additive blending is what produces the glow
//  where many particles overlap.
// -----------------------------------------------------------------------------
inline constexpr const char* kPointVertex = R"GLSL(
#version 330 core
layout(location = 0) in vec4 aPos;    // xy = position, z = speed
layout(location = 1) in vec4 aCol;    // rgba, 0..1

uniform float uInvAspect;             // height / width
uniform float uPointSize;             // base size in pixels

out vec4 vCol;

void main() {
    gl_Position  = vec4(aPos.x * uInvAspect, aPos.y, 0.0, 1.0);
    gl_PointSize = uPointSize * (1.0 + aPos.z);
    vCol = aCol;
}
)GLSL";

inline constexpr const char* kPointFragment = R"GLSL(
#version 330 core

in vec4 vCol;
out vec4 fragColor;

void main() {
    // gl_PointCoord goes 0..1 across the point sprite; make a round, soft dot.
    vec2 d = gl_PointCoord - vec2(0.5);
    float alpha = smoothstep(0.25, 0.0, dot(d, d));
    fragColor = vec4(vCol.rgb, vCol.a * alpha);
}
)GLSL";

// -----------------------------------------------------------------------------
//  Trails. Instead of clearing the screen every frame — which would make the
//  motion look like a slideshow — we draw a single full-screen black triangle
//  with a *tiny* alpha. With normal alpha blending the framebuffer is
//  multiplied by (1 - alpha), so old pixels fade out gradually and the
//  particles leave ribbons.
//
//  Note there are no vertex attributes at all: the triangle is generated from
//  gl_VertexID. That is why this pass only needs a dummy VAO bound.
// -----------------------------------------------------------------------------
inline constexpr const char* kFadeVertex = R"GLSL(
#version 330 core
void main() {
    // ids 0,1,2 -> (-1,-1), (3,-1), (-1,3): one triangle covering the screen
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

inline constexpr const char* kFadeFragment = R"GLSL(
#version 330 core
uniform float uFade;                  // 0 = keep everything, 1 = instant clear
out vec4 fragColor;
void main() { fragColor = vec4(0.0, 0.0, 0.0, uFade); }
)GLSL";
