
#pragma once

#include "pch.h"

namespace pingpong
{
    using Clock = std::chrono::high_resolution_clock;

    using vec2 = glm::vec2;
    using ivec2 = glm::ivec2;
    using vec4 = glm::vec4;
    using mat3 = glm::mat3;

    using PressedKeySet = std::bitset<SDL_NUM_SCANCODES>;

    struct BoardState {
        int match;
        int time;

        bool terminal;
        int winner;
        std::array<float, 2> scores;
        std::array<float, 2> penalties;

        struct Ball {
            vec2 pos;
            vec2 vel;
            float angularVel;
        } ball;
        struct Racquet {
            struct Grip {
                vec2 pos;
                vec2 vel;
            } grip;
            struct Face {
                vec2 pos;
                vec2 vel;
                float angle;
                float angularVel;
            } face;
        };
        std::array<Racquet, 2> racquets;
    };

    class IPlayer
    {
        const int _playerSlot;
    public:
        IPlayer(int slot) : _playerSlot{slot} {}
        virtual ~IPlayer() = default;

        int Slot() const { return _playerSlot; }

        virtual void NewMatch() = 0;
        virtual uint8_t ChooseAction(const BoardState& state, const PressedKeySet& keys) = 0;
        virtual void ProvideFeedback(const BoardState& stateNext) = 0;
    };

} // namespace pingpong
