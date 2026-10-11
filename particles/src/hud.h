// A tiny 5x7 bitmap font rendered as triangles: no font/UI dependencies.
#pragma once
#include <vector>

struct Hud {
  struct Vertex { float x, y, r, g, b, a; };
  std::vector<Vertex> vertices;
  GLuint program = 0, vao = 0, vbo = 0;

  void init() {
    program = link_program(R"GLSL(
#version 330 core
layout(location=0) in vec2 aPos;
layout(location=1) in vec4 aCol;
uniform vec2 uResolution;
out vec4 vCol;
void main() {
  gl_Position = vec4(aPos / uResolution * vec2(2,-2) + vec2(-1,1), 0, 1);
  vCol = aCol;
}
)GLSL", R"GLSL(
#version 330 core
in vec4 vCol;
out vec4 fragColor;
void main() { fragColor = vCol; }
)GLSL");
    glGenVertexArrays(1, &vao); glGenBuffers(1, &vbo);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), nullptr);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)(2 * sizeof(float)));
  }
  void rect(float x, float y, float w, float h, float r, float g, float b, float a = 1) {
    for (int i : {0, 1, 2, 0, 2, 3}) {
      vertices.push_back({x + ((i == 1 || i == 2) ? w : 0),
                          y + (i >= 2 ? h : 0), r, g, b, a});
    }
  }
  void text(float x, float y, const char* s, float scale, float r, float g, float b) {
    static const char* alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./-+ :";
    static const char* glyphs[] = {
      "01110100011000111111100011000110001", "11110100011000111110100011000111110",
      "01111100001000010000100001000001111", "11110100011000110001100011000111110",
      "11111100001000011110100001000011111", "11111100001000011110100001000010000",
      "01111100001000010111100011000101111", "10001100011000111111100011000110001",
      "11111001000010000100001000010011111", "00111000100001000010100101001001100",
      "10001100101010011000101001001010001", "10000100001000010000100001000011111",
      "10001110111010110101100011000110001", "10001110011010110011100011000110001",
      "01110100011000110001100011000101110", "11110100011000111110100001000010000",
      "01110100011000110001101011001001101", "11110100011000111110101001001010001",
      "01111100001000001110000010000111110", "11111001000010000100001000010000100",
      "10001100011000110001100011000101110", "10001100011000110001100010101000100",
      "10001100011000110101101011101110001", "10001100010101000100010101000110001",
      "10001100010101000100001000010000100", "11111000010001000100010001000011111",
      "01110100011001110101110011000101110", "00100011000010000100001000010001110",
      "01110100010000100010001000100011111", "11110000010000101110000010000111110",
      "00010001100101010010111110001000010", "11111100001000011110000010000111110",
      "01110100001000011110100011000101110", "11111000010001000100010000100001000",
      "01110100011000101110100011000101110", "01110100011000101111000010000101110",
      "00000000000000000000000000010000100", "00001000100001000100010000100010000",
      "00000000000000011111000000000000000", "00000001000010011111001000010000000",
      "00000000000000000000000000000000000", "00000001000010000000001000010000000"
    };
    for (; *s; ++s, x += 6 * scale) {
      const char* found = std::strchr(alphabet, *s);
      if (!found) continue;
      const char* bits = glyphs[found - alphabet];
      for (int row = 0; row < 7; ++row)
        for (int col = 0; col < 5; ++col)
          if (bits[row * 5 + col] == '1') rect(x + col * scale, y + row * scale, scale, scale, r, g, b);
    }
  }
  void draw(int w, int h, const char* scene, int count, bool demo, bool paused, bool staged) {
    vertices.clear();
    const float scale = static_cast<float>(h) / 720.0f;
    rect(24 * scale, 24 * scale, 3 * scale, 64 * scale, 0.2f, 0.85f, 1);
    text(40 * scale, 26 * scale, "CUDA / PARTICLE ATLAS", 2 * scale, 0.65f, 0.78f, 0.87f);
    text(40 * scale, 52 * scale, scene, 4 * scale, 0.9f, 0.96f, 1);
    char status[160];
    std::snprintf(status, sizeof status, "%d PARTICLES / %s / %s", count,
                  staged ? "CUDA + COPY" : "ZERO COPY", paused ? "PAUSED" : (demo ? "AUTOPILOT" : "LIVE"));
    text(40 * scale, 94 * scale, status, 1.5f * scale, 0.38f, 0.6f, 0.72f);
    text(28 * scale, h - 32 * scale, "1 ORBIT   2 AURORA   3 INK    A AUTO    C COLOUR    B BLOOM    H HUD", 1.5f * scale, 0.43f, 0.62f, 0.72f);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(program);
    glUniform2f(glGetUniformLocation(program, "uResolution"), w, h);
    glBindVertexArray(vao); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), vertices.data(), GL_STREAM_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(vertices.size()));
  }
  void destroy() {
    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao); glDeleteProgram(program);
  }
};
