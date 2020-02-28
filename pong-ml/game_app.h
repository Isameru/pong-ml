
#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pong
{
    class BoardRenderer;
    class BoardSim;
    class DQEngine;

    /*class PassivePlayer : public IPlayer
    {
    public:
        PassivePlayer(int slot) : IPlayer{slot} {}
        virtual ~PassivePlayer() override = default;
        virtual void NewMatch() override {};
        virtual uint8_t ChooseAction(const BoardState& state, const PressedKeySet& keys) override { return 4; }
        virtual void ProvideFeedback(const BoardState& stateNext) override {}
    };

    class HumanPlayer : public IPlayer
    {
    public:
        HumanPlayer(int slot) : IPlayer{slot} {}
        virtual ~HumanPlayer() override = default;
        virtual void NewMatch() override {};
        virtual uint8_t ChooseAction(const BoardState& state, const PressedKeySet& keys) override;
        virtual void ProvideFeedback(const BoardState& stateNext) override {}
    };*/

    class GameApp
    {
        const bool _vsync;
        const Clock::duration _tickInterval;
        Clock::time_point _startTime;
        Clock::time_point _tickStartTime;
        uint64_t _tickCount = 0;
        bool _quitRequested = false;

        std::unique_ptr<BoardRenderer> _boardRenderer;
        std::unique_ptr<BoardSim> _boardSim;
        std::unique_ptr<DQEngine> _dqEngine;
        std::array<std::unique_ptr<IPlayer>, 2> _players;

        PressedKeySet _pressedKeys;

    public:
        GameApp(bool vsync);
        ~GameApp();
        void Run();

    private:
        static Clock::duration ComputeTickInterval();
        void HandleEvent(const SDL_Event& generalEvent);
        void HandleKeyboardEvent(const SDL_KeyboardEvent& keyboardEvent);
        void OnTick();
        void Draw();
        void Update();
        void NewGame();
    };

} // namespace pong
