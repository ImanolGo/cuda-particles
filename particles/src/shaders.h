// -----------------------------------------------------------------------------
//  shaders.h — point sprites, trail fade, bloom, and presentation
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
uniform float uIntensity;
out vec4 fragColor;

void main() {
    // gl_PointCoord goes 0..1 across the point sprite; make a round, soft dot.
    vec2 d = gl_PointCoord - vec2(0.5);
    // Reversed smoothstep edges are undefined in GLSL; subtract instead.
    float alpha = exp(-dot(d, d) * 24.0) * (1.0 - smoothstep(0.16, 0.25, dot(d, d)));
    fragColor = vec4(vCol.rgb, vCol.a * alpha * uIntensity);
}
)GLSL";

// Separable Gaussian blur, run horizontally then vertically at quarter size.
inline constexpr const char* kBlurFragment = R"GLSL(
#version 330 core
uniform sampler2D uImage;
uniform vec2 uDirection;
uniform vec2 uResolution;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    vec2 stepUV = uDirection / vec2(textureSize(uImage, 0));
    vec3 c = texture(uImage, uv).rgb * 0.227027;
    c += (texture(uImage, uv + stepUV * 1.384615).rgb +
          texture(uImage, uv - stepUV * 1.384615).rgb) * 0.316216;
    c += (texture(uImage, uv + stepUV * 3.230769).rgb +
          texture(uImage, uv - stepUV * 3.230769).rgb) * 0.070270;
    fragColor = vec4(c, 1.0);
}
)GLSL";

inline constexpr const char* kPresentFragment = R"GLSL(
#version 330 core
uniform sampler2D uImage;
uniform sampler2D uBloom;
uniform vec2 uResolution;
uniform float uExposure;
uniform float uGlow;
out vec4 fragColor;
void main() {
    vec2 uv = gl_FragCoord.xy / uResolution;
    vec3 light = texture(uImage, uv).rgb + texture(uBloom, uv).rgb * uGlow;
    // Exponential tone mapping preserves colour instead of clipping to white.
    vec3 c = 1.0 - exp(-light * uExposure);
    float vignette = 1.0 - 0.48 * dot(uv - 0.5, uv - 0.5);
    c = pow(c, vec3(0.8)) * vignette;
    c += vec3(0.008, 0.014, 0.030) * (0.6 + 0.4 * (1.0 - uv.y));
    fragColor = vec4(c, 1.0);
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
