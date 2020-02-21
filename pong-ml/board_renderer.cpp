
#include "pch.h"
#include "board_renderer.h"

#include "config.h"

namespace pingpong
{
    BoardRenderer::BoardRenderer(bool vsync)
    {
        const auto boardScreenSize = ivec2{cfg.boardScreenWidth, 2 * cfg.boardScreenWidth};
        const auto screenSize = ivec2{boardScreenSize.x + 2 * cfg.boardScreenMargin, boardScreenSize.y + 2 * cfg.boardScreenMargin};

        _window = SDL_CreateWindow(GameWindowTitle, SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screenSize.x, screenSize.y, SDL_WINDOW_OPENGL);
        _context = SDL_GL_CreateContext(_window);

        glewExperimental = GL_TRUE;
        GLenum glewError = glewInit();

        if (glewError != GLEW_OK)
            throw std::runtime_error(std::string{"unable to initialize the OpenGL Extension Wrangler Library: glewInit() failed: "} + reinterpret_cast<const char*>(glewGetErrorString(glewError)));

        SDL_GL_SetSwapInterval(vsync ? 1 : 0);

        _shaderProgramId = MakeShaderProgram();
        _transformLocation = glGetUniformLocation(_shaderProgramId, "transform");
        _colorLocation = glGetUniformLocation(_shaderProgramId, "color");

        _rectVertexBufferId = MakeRectVertexBuffer(vec2(1.0f, 1.0f));
        _circleVertexBufferId = MakeCircleVertexBuffer(1.0f);

        const auto proportion = static_cast<float>(boardScreenSize.x) / static_cast<float>(boardScreenSize.y);

        const auto scale = vec2{
            static_cast<float>(boardScreenSize.x) / static_cast<float>(screenSize.x),
            static_cast<float>(boardScreenSize.y) / static_cast<float>(screenSize.y) };

        _view = glm::mat3(
            scale.x * 2.0f, 0.0f, 0.0f,
            0.0f, scale.y * 2.0f * proportion, 0.0f,
            0.0f, 0.0f, 1.0f);
        _boardRectTransform = glm::mat3(
            1.0f, 0.0f, 0.0f,
            0.0f, 2.0f, 0.0f,
            0.0f, 0.0f, 1.0f) * _view;
    }

    BoardRenderer::~BoardRenderer()
    {
        // TODO
    }

    void BoardRenderer::Draw(const BoardState& boardState)
    {
        glClearColor(cfg.backgroundColor.r, cfg.backgroundColor.g, cfg.backgroundColor.b, cfg.backgroundColor.a);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(_shaderProgramId);
        glEnableVertexAttribArray(0);

        DrawBoardRect(cfg.secondaryColor);
        DrawCircle(boardState.ball.pos, cfg.ballRadius, cfg.primaryColor);

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            DrawCircle(boardState.racquets[playerSlot].grip.pos, cfg.racquetGripRadius, cfg.secondaryColor);
            DrawRect(boardState.racquets[playerSlot].face.pos, cfg.racquetFaceDim, boardState.racquets[playerSlot].face.angle, cfg.primaryColor);
        }

        SDL_GL_SwapWindow(_window);
    }

    BoardRenderer::ShaderProgramId BoardRenderer::MakeShaderProgram()
    {
        const char* vertexShader =
            "#version 330 core\n"
            "layout(location = 0) in vec2 pos;\n"
            "uniform mat2x3 transform;\n"
            "void main() {\n"
            "gl_Position = vec4(vec3(pos, 1.0) * transform, 0.0, 1.0); }";
        const char* fragmentShader =
            "#version 330 core\n"
            "uniform vec4 color;\n"
            "out vec4 out_color;\n"
            "void main() {\n"
            "out_color = color; }";
        //"out_color = vec4(1.0, 1.0, 1.0, 1.0); }";

        GLuint vertexShaderId = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertexShaderId, 1, &vertexShader, nullptr);
        glCompileShader(vertexShaderId);

        GLuint fragmentShaderId = glCreateShader(GL_FRAGMENT_SHADER);
        glShaderSource(fragmentShaderId, 1, &fragmentShader , nullptr);
        glCompileShader(fragmentShaderId);

        GLuint shaderProgramId = glCreateProgram();
        glAttachShader(shaderProgramId, vertexShaderId);
        glAttachShader(shaderProgramId, fragmentShaderId);
        glLinkProgram(shaderProgramId);

        glDetachShader(shaderProgramId, vertexShaderId);
        glDetachShader(shaderProgramId, fragmentShaderId);
        glDeleteShader(vertexShaderId);
        glDeleteShader(fragmentShaderId);

        return shaderProgramId;
    }

    BoardRenderer::VertexBufferId BoardRenderer::MakeRectVertexBuffer(vec2 size)
    {
        GLfloat vertices[] = {
            -0.5f * size.x, -0.5f * size.y,
            -0.5f * size.x, +0.5f * size.y,
            +0.5f * size.x, +0.5f * size.y,
            +0.5f * size.x, -0.5f * size.y };

        GLuint vertexBufferId;
        glGenBuffers(1, &vertexBufferId);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
        glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(GLfloat), &vertices[0], GL_STATIC_DRAW);

        return vertexBufferId;
    }

    BoardRenderer::VertexBufferId BoardRenderer::MakeCircleVertexBuffer(float radius)
    {
        std::vector<vec2> vertices;
        vertices.reserve(_circleVertexCount);

        vertices.emplace_back(0.0f, 0.0f);

        const float pi = std::acos(-1.0f);
        const int stepCount = _circleVertexCount - 2;
        const float alpha = 2 * pi / static_cast<float>(stepCount);
        for (int i = 0; i < stepCount; ++i)
        {
            const float angle = alpha * static_cast<float>(i);
            vertices.emplace_back(radius * -std::cos(angle), radius * std::sin(angle));
        }
        vertices.push_back(vertices[1]);

        GLuint vertexBufferId;
        glGenBuffers(1, &vertexBufferId);
        glBindBuffer(GL_ARRAY_BUFFER, vertexBufferId);
        glBufferData(GL_ARRAY_BUFFER, static_cast<size_t>(_circleVertexCount) * 2 * sizeof(GLfloat), reinterpret_cast<const GLfloat*>(&vertices[0]), GL_STATIC_DRAW);

        return vertexBufferId;
    }

    void BoardRenderer::DrawCircle(vec2 pos, float radius, vec4 color)
    {
        glBindBuffer(GL_ARRAY_BUFFER, _circleVertexBufferId);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        const auto transform = glm::mat3(
            radius, 0.0f, pos.x,
            0.0f, radius, pos.y,
            0.0f, 0.0f, 1.0f) * _view;
        glUniformMatrix2x3fv(_transformLocation, 1, GL_FALSE, &transform[0][0]);
        glUniform4fv(_colorLocation, 1, &color[0]);
        glDrawArrays(GL_TRIANGLE_FAN, 0, _circleVertexCount);
    }

    void BoardRenderer::DrawRect(vec2 pos, vec2 dim, float rotation, vec4 color)
    {
        glBindBuffer(GL_ARRAY_BUFFER, _rectVertexBufferId);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        const auto transform =
            glm::mat3(
                dim.x, 0.0f, 0.0f,
                0.0f, dim.y, 0.0f,
                0.0f, 0.0f, 1.0f) *
            glm::mat3(
                std::cos(rotation), -std::sin(rotation), pos.x,
                std::sin(rotation), std::cos(rotation), pos.y,
                0.0f, 0.0f, 1.0f) *
            _view;
        glUniformMatrix2x3fv(_transformLocation, 1, GL_FALSE, &transform[0][0]);
        glUniform4fv(_colorLocation, 1, &color[0]);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }

    void BoardRenderer::DrawBoardRect(vec4 color)
    {
        glBindBuffer(GL_ARRAY_BUFFER, _rectVertexBufferId);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glUniformMatrix2x3fv(_transformLocation, 1, GL_FALSE, &_boardRectTransform[0][0]);
        glUniform4fv(_colorLocation, 1, &color[0]);
        glDrawArrays(GL_LINE_LOOP, 0, 4);
    }

} // namespace pingpong
