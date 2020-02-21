
// Note: DQ stands for Deep Q-Learning - a reinforced learning algorithm.

#pragma once

#include "pch.h"
#include "basic_types.h"

namespace pingpong
{
    extern const torch::Device LibTorchDevice;

    struct Batch
    {
        torch::Tensor state;
        torch::Tensor action;
        torch::Tensor stateNext;
        torch::Tensor reward;
        torch::Tensor nonTerminal;
    };

    class ReplayMemory
    {
        const int _capacity;
        int _size = 0;
        int _index = 0;

        std::vector<float> _stateData;
        std::vector<int64_t> _actionData;
        std::vector<float> _stateNextData;
        std::vector<float> _rewardData;
        std::vector<uint8_t> _nonTerminalData;

        torch::Tensor _stateTensor;
        torch::Tensor _actionTensor;
        torch::Tensor _stateNextTensor;
        torch::Tensor _rewardTensor;
        torch::Tensor _nonTerminalTensor;

    public:
        ReplayMemory(int capacity);
        void Memorize(const std::vector<float>& state, uint8_t action, const std::vector<float>& stateNext, float reward, bool terminal);
        Batch Sample(std::mt19937& randGen, int batchSize);
        int Size() const noexcept { return _size; }
    };

    struct DQNet : torch::nn::Module
    {
        torch::nn::Linear layer_0;
        torch::nn::Linear layer_1;
        torch::nn::Linear layer_2;
        torch::nn::Linear layer_3;
        torch::nn::LeakyReLU leakyReLU;

        DQNet();
        torch::Tensor forward(torch::Tensor input);
    };

    class DQEngine
    {
        const std::string _modelPath;
        DQNet _net;
        torch::optim::Adam _optimizer;
        ReplayMemory _replayMemory;
        std::mt19937 _randGen;

    public:
        DQEngine(std::string modelPath);
        void NewMatch();
        uint8_t ChooseAction(const std::vector<float>& stateVector);
        void Memorize(const std::vector<float>& stateVector, uint8_t action, const std::vector<float>& stateNextVector, float reward, bool terminal);
        void Optimize();
    };

    class DQBot : public IPlayer
    {
        DQEngine& _dqEngine;
        std::mt19937 _randGen;
        float _randomPolicyThreshold {};
        std::vector<float> _lastState;
        uint8_t _lastAction {};
        bool _humanControl {};
        bool _noRandomPolicy = false;

    public:
        DQBot(int slot, DQEngine& dqEngine);
        virtual ~DQBot() override = default;
        virtual void NewMatch() override;
        virtual uint8_t ChooseAction(const BoardState& state, const PressedKeySet& keys) override;
        virtual void ProvideFeedback(const BoardState& stateNext) override;
    };

} // namespace pingpong
