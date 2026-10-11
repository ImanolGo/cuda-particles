// Persistent HDR trails + two quarter-resolution bloom targets.
// No feedback loop: fading uses blending, never samples the attached texture.
#pragma once

struct RenderTarget {
  GLuint fbo = 0, texture = 0;
  int width = 0, height = 0;

  void destroy() {
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &texture);
    fbo = texture = 0;
  }
  void resize(int w, int h) {
    if (width == w && height == h && fbo) return;
    destroy();
    width = w; height = h;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, w, h, 0, GL_RGBA, GL_FLOAT, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      std::fprintf(stderr, "HDR framebuffer is incomplete\n");
      std::exit(EXIT_FAILURE);
    }
    clear();
  }
  void bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, width, height);
  }
  void clear() const {
    bind();
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
  }
};

struct Renderer {
  RenderTarget trails, bloom[2];
  GLuint blur = 0, present = 0;

  void init() {
    blur = link_program(kFadeVertex, kBlurFragment);
    present = link_program(kFadeVertex, kPresentFragment);
  }
  void resize(int w, int h) {
    trails.resize(w, h);
    bloom[0].resize(w / 4 > 0 ? w / 4 : 1, h / 4 > 0 ? h / 4 : 1);
    bloom[1].resize(bloom[0].width, bloom[0].height);
  }
  void draw(GLuint empty_vao, float exposure, bool glow) const {
    glDisable(GL_BLEND);
    glBindVertexArray(empty_vao);
    glActiveTexture(GL_TEXTURE0);
    glUseProgram(blur);
    for (int pass = 0; glow && pass < 2; ++pass) {
      bloom[pass].bind();
      glBindTexture(GL_TEXTURE_2D, pass == 0 ? trails.texture : bloom[0].texture);
      glUniform2f(glGetUniformLocation(blur, "uResolution"), bloom[pass].width, bloom[pass].height);
      glUniform2f(glGetUniformLocation(blur, "uDirection"), pass == 0 ? 4.0f : 0.0f,
                  pass == 0 ? 0.0f : 1.0f);
      glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, trails.width, trails.height);
    glUseProgram(present);
    glBindTexture(GL_TEXTURE_2D, trails.texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, bloom[1].texture);
    glUniform1i(glGetUniformLocation(present, "uBloom"), 1);
    glUniform2f(glGetUniformLocation(present, "uResolution"), trails.width, trails.height);
    glUniform1f(glGetUniformLocation(present, "uExposure"), exposure);
    glUniform1f(glGetUniformLocation(present, "uGlow"), glow ? 0.85f : 0.0f);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glActiveTexture(GL_TEXTURE0);
    glEnable(GL_BLEND);
  }
  void destroy() {
    trails.destroy(); bloom[0].destroy(); bloom[1].destroy();
    glDeleteProgram(blur); glDeleteProgram(present);
  }
};
