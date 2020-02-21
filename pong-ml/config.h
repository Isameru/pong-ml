
#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pingpong
{
    struct Config
    {
        int boardScreenWidth = 448;   // Height is (always) twice as that.
        int boardScreenMargin = 10;

        vec4 primaryColor = {1.0f, 1.0f, 1.0f, 1.0f};
        vec4 secondaryColor = {0.5f, 0.5f, 0.5f, 1.0f};
        vec4 backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};

        double timeStep = 1.0 / 30.0;
        int matchTickLimit = 2000;
        float wallsElasticity = 0.45f;
        float ballRadius = 0.026f;
        float ballMass = 0.35f;
        float ballElasticity = 0.8f;
        float ballMotionResistance = 0.45f;
        float racquetToCenterDistance = 0.85f;
        float racquetGripRadius = 0.025f;
        float racquetGripMass = 2.0f;
        float racquetGripElasticity = 0.0f;
        float racquetGripSteeringForce = 1.0f;
        vec2 racquetFaceDim = {0.0175, 0.18};
        float racquetFaceMass = 1.5f;
        float racquetFaceElasticity = 0.75f;
        float racquetFaceMotionResistance = 0.45f;

        int replayMemoryCapacity = 8 * 1024;   // How much last transitions made shall be kept in the replay memory.
        int sampleBatchSize = 128;     // How much transitions shall be recalled with every time step to propagate the rewards back through time and space of possible states.
        float gamma = 0.999f;   // The discount factor - the effective loss of reward each time it is propagated back through time and space of possible states.
    };

    extern Config cfg;  // Instantiated in main.cpp

} // namespace pingpong
