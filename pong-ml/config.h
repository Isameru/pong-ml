
#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pong
{
    struct Config
    {
        // Command Line Options
        //
        std::vector<std::string_view> args;
        bool printSynopsis = false;
        bool vsync = true;
        std::string modelPath;
        bool verbose = false;

        // Display
        //
        int boardScreenWidth = 448;   // Height is (always) twice as that.
        int boardScreenMargin = 10;

        vec4 primaryColor = {1.0f, 1.0f, 1.0f, 1.0f};
        vec4 secondaryColor = {0.5f, 0.5f, 0.5f, 1.0f};
        vec4 backgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};

        // Simulation
        //
        double timeStep = 1.0 / 12.0;
        int subSteps = 5;
        int matchTickLimit = 5000;
        float gravity = 0.175f;
        float wallsElasticity = 0.45f;
        float ballRadius = 0.026f;
        float ballMass = 0.35f;
        float ballElasticity = 0.8f;
        float ballMotionResistance = 0.15f;
        float racquetToCenterDistance = 0.85f;
        float racquetGripRadius = 0.025f;
        float racquetGripMass = 2.0f;
        float racquetGripElasticity = 0.0f;
        float racquetGripSteeringForce = 1.85f;
        vec2 racquetFaceDim = {0.0175, 0.18};
        float racquetFaceMass = 1.5f;
        float racquetFaceElasticity = 0.75f;
        float racquetFaceMotionResistance = 0.15f;

        // Rewards
        //
        float winScore = 0.0f;
        float ballInterceptionScore = 1.0f;
        float ballSendingScoreToSpeed = 1.0f;

        // Machine Learning
        //
        torch::Device libTorchDevice = torch::kCPU;
        int syncNetsAfterMatches = 50;
        int replayMemoryCapacity = 16 * 1024 * 1024;  // How much last transitions made shall be kept in the replay memory.
        int minMemoriesToLearn = 2 * 1024 * 1024;
        int optimizeCountLimit = 1024;
        int sampleBatchSize = 32;   // How much transitions shall be recalled with every time step to propagate the rewards back through time and space of possible states.
        float learningRate = 1e-4;
        float gamma = 0.9975f;   // The discount factor - the effective loss of reward each time it is propagated back through time and space of possible states.
    };

    extern Config cfg;  // Convenience global variable singleton; instantiated in main.cpp, where it is altered by the command line parser during startup.

} // namespace pong
