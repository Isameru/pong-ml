
#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pong
{
    constexpr char GameWindowTitle[] = "Pong :: The Machine Learning Game";

    class BoardRenderer
    {
        using ShaderProgramId = GLuint;
        using VertexBufferId = GLuint;
        using UniformLocationId = GLint;

        SDL_Window* _window = nullptr;
        SDL_GLContext _context {};

        ShaderProgramId _shaderProgramId {};
        UniformLocationId _transformLocation {};
        UniformLocationId _colorLocation {};
        VertexBufferId _rectVertexBufferId {};
        const int _circleVertexCount = 32;
        VertexBufferId _circleVertexBufferId {};

        mat3 _view;
        mat3 _boardRectTransform;

    public:
        BoardRenderer(bool vsync);
        ~BoardRenderer();
        void Draw(const BoardState& boardState);

    private:
        ShaderProgramId MakeShaderProgram();
        VertexBufferId MakeRectVertexBuffer(vec2 size);
        VertexBufferId MakeCircleVertexBuffer(float radius);

        void DrawCircle(vec2 pos, float radius, vec4 color);
        void DrawRect(bool filled, vec2 pos, vec2 dim, vec4 color);
        void DrawRect(bool filled, vec2 pos, vec2 dim, float rotation, vec4 color);
        void DrawRect(bool filled, mat3 transform, vec4 color);
    };

} // namespace pong
