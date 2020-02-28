
#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pong
{
    class BoardSim
    {
        struct CollisionType {
            constexpr static cpCollisionType Ball = 1;
            constexpr static cpCollisionType Racquet = 2;
        };

        cpSpace* _space {};
        cpBody* _ballBody {};
        float _lastBallVelY {};
        struct Racquet {
            cpBody* gripBody {};
            cpShape* gripShape {};
            cpBody* faceBody {};
            cpShape* faceShape {};
        };
        std::array<Racquet, 2> _racquets;

        BoardState _state;

    public:
        BoardSim();
        ~BoardSim();

        void NewMatch();
        void Step(std::array<uint8_t, 2> actions);
        const BoardState& State() { assert(_state.match >= 0 && "Call BoardSim::NewMatch() first"); return _state; }

    private:
        void UpdateState();
        void CheckTerminalCondition();
        void ResetSpace();

        static void BallRacquetCollisionPostSolveCallback(cpArbiter* arbiter, cpSpace* space, void* data);
        void OnBallRacquetCollisionPostSolve(cpArbiter* arbiter);
    };

} // namespace pong
