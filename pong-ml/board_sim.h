
#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pingpong
{
    class BoardSim
    {
        cpSpace* _space {};
        cpBody* _ballBody {};
        struct Racquet {
            cpBody* gripBody {};
            cpBody* faceBody {};
            cpConstraint* pivotJoint = nullptr;
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
    };

} // namespace pingpong
