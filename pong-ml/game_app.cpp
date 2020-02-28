
#include "pch.h"
#include "game_app.h"

#include "board_renderer.h"
#include "board_sim.h"
#include "config.h"
#include "dq_bot.h"

namespace pong
{
    /*uint8_t HumanPlayer::ChooseAction(const BoardState& state, const PressedKeySet& keys)
    {
        int move_x = 0;

        if (Slot() == 0)
        {
            if (keys.test(SDL_SCANCODE_LEFT)) move_x -= 1;
            if (keys.test(SDL_SCANCODE_RIGHT)) move_x += 1;
        }
        else if (Slot() == 1)
        {
            // Note: For the second (upper) slot, effective "left" and "right" controls are swapped.
            if (keys.test(SDL_SCANCODE_A)) move_x += 1;
            if (keys.test(SDL_SCANCODE_D)) move_x -= 1;
        }
        else assert(false);

        if (move_x == -1) {
            return 0;
        }
        else if (move_x == 1) {
            return 1;
        }
        else if (move_x == 0) {
            return state.racquets[Slot()].grip.vel.x > 0.0f ? Slot() : 1 - Slot();
        }
        else assert(false);
    }*/

    GameApp::GameApp(bool vsync) :
        _vsync{vsync},
        _tickInterval{ComputeTickInterval()}
    {
        SDL_Init(SDL_INIT_VIDEO);
        //IMG_Init(IMG_INIT_PNG);
        //TTF_Init();

        _boardRenderer = std::make_unique<BoardRenderer>(_vsync);
        _boardSim = std::make_unique<BoardSim>();

        _dqEngine = std::make_unique<DQEngine>(cfg.modelPath);

        _players[0] = std::make_unique<DQBot>(0, *_dqEngine);
        _players[1] = std::make_unique<DQBot>(1, *_dqEngine);

        NewGame();
    }

    GameApp::~GameApp()
    {
        //TTF_Quit();
        //IMG_Quit();
        SDL_Quit();
    }

    void GameApp::Run()
    {
        _startTime = Clock::now();
        _tickStartTime = {};

/*#ifdef __EMSCRIPTEN__
        emscripten_set_main_loop_arg([](void* ctx) {
            reinterpret_cast<GameApp*>(ctx)->OnTick();
        }, this, 0, 1);
#else*/

        while (!_quitRequested)
        {
            if (_vsync && _tickStartTime != Clock::time_point{}) {
                while (Clock::now() - _tickStartTime < _tickInterval) {
                    std::this_thread::yield();
                }
            }
            _tickStartTime = Clock::now();

            OnTick();
            ++_tickCount;

            /*if (_tickCount % 60 == 0) {
                const auto elapsedMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - _startTime).count();
                std::cout << "FPS: " << 1000.0 * static_cast<double>(_tickCount) / static_cast<double>(elapsedMilliseconds) << std::endl;
            }*/
        }
    }

    Clock::duration GameApp::ComputeTickInterval()
    {
        return std::chrono::nanoseconds{static_cast<int64_t>(cfg.timeStep * 1e9)};
    }

    void GameApp::HandleEvent(const SDL_Event& generalEvent)
    {
        switch (generalEvent.type)
        {
        case SDL_KEYDOWN:
        case SDL_KEYUP:
            HandleKeyboardEvent(reinterpret_cast<const SDL_KeyboardEvent&>(generalEvent));
            break;
        case SDL_QUIT:
            _quitRequested = true;
            break;
        }
    }

    void GameApp::HandleKeyboardEvent(const SDL_KeyboardEvent& keyboardEvent)
    {
        _pressedKeys.set(static_cast<size_t>(keyboardEvent.keysym.scancode), keyboardEvent.type == SDL_KEYDOWN);
    }

    void GameApp::OnTick()
    {
        SDL_Event event {};
        while (SDL_PollEvent(&event))
            HandleEvent(event);

        if (_quitRequested)
            return;

        Draw();
        Update();
    }

    void GameApp::Draw()
    {
        _boardRenderer->Draw(_boardSim->State());
    }

    void GameApp::Update()
    {
        const auto& state = _boardSim->State();

        if (state.terminal)
        {
            assert(state.winner >= -1 && state.winner <= 1);
            std::cout << "Match " << state.match << " conluded after " << state.time - 1 << " ticks: ";
            if (state.winner == -1)
                std::cout << "Draw";
            else
                std::cout << "Player " << state.winner << " wins";
            std::cout << std::endl;

            NewGame();
        }
        else
        {
            auto actions = std::array<uint8_t, 2>{
                _players[0]->ChooseAction(state, _pressedKeys),
                _players[1]->ChooseAction(state, _pressedKeys)
            };

            _boardSim->Step(actions);

            const auto& stateNext = _boardSim->State();
            _players[0]->ProvideFeedback(stateNext);
            _players[1]->ProvideFeedback(stateNext);

            //_dqEngine->Optimize();
        }
    }

    void GameApp::NewGame()
    {
        _boardSim->NewMatch();
        _dqEngine->NewMatch();   // Network optimization takes place here.
        _players[0]->NewMatch();
        _players[1]->NewMatch();

        std::cout << "Match " << _boardSim->State().match << " started" << std::endl;
    }

} // namespace pong
