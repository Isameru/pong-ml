
#include "pch.h"
#include "dq_bot.h"

#include "config.h"

namespace pingpong
{
    torch::Device DetermineLibTorchDevice()
    {
        const torch::Device device = torch::cuda::cudnn_is_available() ? torch::kCUDA : torch::kCPU;
        std::cout << "LibTorch CUDA Support: " << (device.is_cuda() ? "Yes" : "No") << std::endl;
        return device;
    }

    const torch::Device LibTorchDevice = DetermineLibTorchDevice();

    template<typename IntT>
    std::vector<IntT> SampleBatchIndices(std::mt19937& randGen, IntT N, IntT K, std::vector<IntT>&& reuse = {})
    {
        using namespace std;

        thread_local std::vector<IntT> shuffleIndices;
        const auto N_uint64 = static_cast<size_t>(N);

        if (shuffleIndices.size() > N_uint64)
            shuffleIndices.clear();

        while (shuffleIndices.size() < N_uint64)
            shuffleIndices.push_back(static_cast<IntT>(shuffleIndices.size()));

        for (IntT k = 0; k < K; ++k)
        {
            const IntT i = std::uniform_int_distribution<IntT>{k, N - 1}(randGen);
            std::swap(shuffleIndices[k], shuffleIndices[i]);
        }

        reuse.clear();
        reuse.insert(begin(reuse), begin(shuffleIndices), begin(shuffleIndices) + K);
        assert(reuse.size() == static_cast<size_t>(K));
        return reuse;
    }

    std::vector<float> MakeStateVector(const BoardState& state, int playerSlot, std::vector<float>&& reuse = {})
    {
        const float sideFactor = 2.0f * static_cast<float>(playerSlot) - 1.0f;

        reuse.clear();
        reuse.reserve(5 + 2 * 11);

        reuse.push_back(sideFactor * state.ball.pos.x);
        reuse.push_back(sideFactor * state.ball.pos.y);
        reuse.push_back(sideFactor * state.ball.vel.x);
        reuse.push_back(sideFactor * state.ball.vel.y);
        reuse.push_back(sideFactor * state.ball.angularVel);

        for (int playerSlot = 0; playerSlot < 2; ++playerSlot)
        {
            reuse.push_back(sideFactor * state.racquets[playerSlot].grip.pos.x);
            reuse.push_back(sideFactor * state.racquets[playerSlot].grip.pos.y);
            reuse.push_back(sideFactor * state.racquets[playerSlot].grip.vel.x);
            reuse.push_back(sideFactor * state.racquets[playerSlot].grip.vel.y);
            reuse.push_back(sideFactor * state.racquets[playerSlot].face.pos.x);
            reuse.push_back(sideFactor * state.racquets[playerSlot].face.pos.y);
            reuse.push_back(sideFactor * state.racquets[playerSlot].face.vel.x);
            reuse.push_back(sideFactor * state.racquets[playerSlot].face.vel.y);
            reuse.push_back(sideFactor * std::sin(state.racquets[playerSlot].face.angle));
            reuse.push_back(sideFactor * std::cos(state.racquets[playerSlot].face.angle));
            reuse.push_back(sideFactor * state.racquets[playerSlot].face.angularVel);
        }

        return reuse;
    }

    float ComputeStateReward(const BoardState& state, int playerSlot, const std::vector<float>& stateVector)
    {
        float reward = state.scores[playerSlot] - state.penalties[playerSlot];
        // if (state.terminal)
        //     reward += (state.winner == playerSlot) ? +50.0f : -50.0f;
        return reward;
    }

    const int InputStateVectorLength = static_cast<int>(MakeStateVector(BoardState{}, 0).size());
    const int OutputVectorLength = 9;

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // ReplayMemory
    //

    ReplayMemory::ReplayMemory(int capacity) :
        _capacity{capacity}
    {
        _stateData.resize(capacity * InputStateVectorLength);
        _actionData.resize(capacity);
        _stateNextData.resize(capacity * InputStateVectorLength);
        _rewardData.resize(capacity);
        _nonTerminalData.resize(capacity);

        _stateTensor = torch::from_blob(_stateData.data(), {_capacity, InputStateVectorLength}, torch::TensorOptions{}.dtype(torch::kFloat32));
        _actionTensor = torch::from_blob(_actionData.data(), {_capacity}, torch::TensorOptions{}.dtype(torch::kInt64));
        _stateNextTensor = torch::from_blob(_stateNextData.data(), {_capacity, InputStateVectorLength}, torch::TensorOptions{}.dtype(torch::kFloat32));
        _rewardTensor = torch::from_blob(_rewardData.data(), {_capacity}, torch::TensorOptions{}.dtype(torch::kFloat32));
        _nonTerminalTensor = torch::from_blob(_nonTerminalData.data(), {_capacity}, torch::TensorOptions{}.dtype(torch::kBool));
    }

    void ReplayMemory::Memorize(const std::vector<float>& state, uint8_t action, const std::vector<float>& stateNext, float reward, bool terminal)
    {
        std::copy(begin(state), end(state), begin(_stateData) + _index * InputStateVectorLength);
        _actionData[_index] = int64_t{action};
        std::copy(begin(stateNext), end(stateNext), begin(_stateNextData) + _index * InputStateVectorLength);
        _rewardData[_index] = reward;
        _nonTerminalData[_index] = static_cast<uint8_t>(!terminal);

        ++_index;
        _size = std::max(_size, _index);
        _index = _index % _capacity;
    }

    Batch ReplayMemory::Sample(std::mt19937& randGen, int batchSize)
    {
        batchSize = std::min(_size, batchSize);
        auto indexData = SampleBatchIndices<int64_t>(randGen, _size, batchSize);
        auto indexTensor = torch::from_blob(indexData.data(), {batchSize}, torch::TensorOptions{}.dtype(torch::kInt64));

        return {
            _stateTensor.index_select(0, indexTensor).to(LibTorchDevice),
            _actionTensor.index_select(0, indexTensor).to(LibTorchDevice),
            _stateNextTensor.index_select(0, indexTensor).to(LibTorchDevice),
            _rewardTensor.index_select(0, indexTensor).to(LibTorchDevice),
            _nonTerminalTensor.index_select(0, indexTensor).to(LibTorchDevice)
        };
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DQNet
    //

    DQNet::DQNet() :
        layer_0{register_module("layer_0", torch::nn::Linear(InputStateVectorLength, 32))},
        layer_1{register_module("layer_1", torch::nn::Linear(32, 32))},
        layer_2{register_module("layer_2", torch::nn::Linear(32, 32))},
        layer_3{register_module("layer_3", torch::nn::Linear(32, OutputVectorLength))},
        leakyReLU{torch::nn::LeakyReLUOptions().negative_slope(0.18)}
    { }

    torch::Tensor DQNet::forward(torch::Tensor x)
    {
        x = leakyReLU(layer_0(x));
        x = leakyReLU(layer_1(x));
        x = leakyReLU(layer_2(x));
        x = layer_3(x);
        return x;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DQEngine
    //

    DQEngine::DQEngine(std::string modelPath) :
        _modelPath{std::move(modelPath)},
        _optimizer{_net.parameters(), torch::optim::AdamOptions{1e-5}.beta1(0.5)},
        _replayMemory{cfg.replayMemoryCapacity},
        _randGen{std::random_device{}()}
    {
        auto inputModelFile = std::ifstream{_modelPath};
        if (inputModelFile.is_open()) {
            std::cout << "Loading model: " << _modelPath << std::endl;
            torch::serialize::InputArchive inputArchive;
            inputArchive.load_from(inputModelFile);
            _net.load(inputArchive);
        }
        _net.to(LibTorchDevice);
    }

    void DQEngine::NewMatch()
    {
        torch::serialize::OutputArchive outputArchive;
        _net.save(outputArchive);
        outputArchive.save_to(_modelPath);
    }

    uint8_t DQEngine::ChooseAction(const std::vector<float>& stateVector)
    {
        auto input = torch::tensor(stateVector, torch::TensorOptions{}.device(LibTorchDevice));
        assert(!input.requires_grad());
        const auto output = _net.forward(input);
        //std::cout << "net(input): " << output.view({1, -1}) << std::endl;
        auto outputAction = std::get<1>(output.max(0)).item().toByte();
        return outputAction;
    }

    void DQEngine::Memorize(const std::vector<float>& stateVector, uint8_t action, const std::vector<float>& stateNextVector, float reward, bool terminal)
    {
        _replayMemory.Memorize(stateVector, action, stateNextVector, reward, terminal);
    }

    void DQEngine::Optimize()
    {
        auto batch = _replayMemory.Sample(_randGen, cfg.sampleBatchSize);

        // Compute the Q-value of the actually exercised action (so called Pi, or "policy" function value) on the predecessor state.
        //
        batch.state.set_requires_grad(true);
        auto Q_state_actionValues = _net.forward(batch.state);
        auto Q_state_chosenActionValue = Q_state_actionValues.gather(1, batch.action.view({-1, 1}));

        // Compute the best possible Q-value which can be exercised from the successor state.
        //
        batch.stateNext.set_requires_grad(false);
        auto Q_stateNext_actionValues = _net.forward(batch.stateNext);
        auto Q_stateNext_bestActionValue = std::get<0>(Q_stateNext_actionValues.max(1));

        // Based on the best Q-value of the successor state, compute the expected value (a value to be learnt) of chosen action on the predecessor state.
        //
        auto expected_Q_state_chosenActionValue = (batch.reward + batch.nonTerminal * cfg.gamma * Q_stateNext_bestActionValue).detach();

        // Compute the loss as a criterion function between the actual Q-values and the expected (future reward discounted by time).
        //
        _optimizer.zero_grad();

        auto loss = torch::l1_loss(Q_state_chosenActionValue, expected_Q_state_chosenActionValue);
        loss.backward();

        _optimizer.step();
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DQBot
    //

    DQBot::DQBot(int slot, DQEngine& dqEngine) :
        IPlayer{slot},
        _dqEngine{dqEngine},
        _randGen{std::random_device{}()}
    { }

    void DQBot::NewMatch()
    {
        _randomPolicyThreshold = std::uniform_real_distribution<float>{}(_randGen);
        _humanControl = false;
    }

    uint8_t DQBot::ChooseAction(const BoardState& state, const PressedKeySet& keys)
    {
        _lastState = MakeStateVector(state, Slot(), std::move(_lastState));

        // Read the keyboard seeking for human interaction.
        //
        auto move = ivec2{0, 0};

        if (Slot() == 0)
        {
            if (keys.test(SDL_SCANCODE_LEFT)) move.x -= 1;
            if (keys.test(SDL_SCANCODE_RIGHT)) move.x += 1;
            if (keys.test(SDL_SCANCODE_DOWN)) move.y -= 1;
            if (keys.test(SDL_SCANCODE_UP)) move.y += 1;
        }
        else if (Slot() == 1)
        {
            // Note: For the second (upper) slot, effective "left" and "right" controls are swapped.
            if (keys.test(SDL_SCANCODE_A)) move.x += 1;
            if (keys.test(SDL_SCANCODE_D)) move.x -= 1;
            if (keys.test(SDL_SCANCODE_W)) move.y -= 1;
            if (keys.test(SDL_SCANCODE_S)) move.y += 1;
        }
        else assert(false);

        bool new_noRandomPolicy = _noRandomPolicy;
        if (keys.test(SDL_SCANCODE_1)) {
            new_noRandomPolicy = true;
        }
        else if (keys.test(SDL_SCANCODE_2)) {
            new_noRandomPolicy = false;
        }

        if (_noRandomPolicy != new_noRandomPolicy) {
            _noRandomPolicy = new_noRandomPolicy;
            std::cout << "Option: no-random-policy: " << (_noRandomPolicy ? "On" : "Off") << std::endl;
        }

        uint8_t humanAction = 3 * (move.y + 1) + (move.x + 1);

        if (humanAction != 4)
            _humanControl = true;

        // Determine the action basen on either human interaction, random policy, or best Q-value.
        //
        if (_humanControl)
        {
            _lastAction = humanAction;
        }
        else
        {
            const float rf = std::uniform_real_distribution<float>{}(_randGen);

            if (!_noRandomPolicy && rf <= _randomPolicyThreshold) {
                _lastAction = std::uniform_int_distribution<uint8_t>{0, OutputVectorLength - 1}(_randGen);
            }
            else {
                _lastAction = _dqEngine.ChooseAction(_lastState);
            }
        }

        return _lastAction;
    }

    void DQBot::ProvideFeedback(const BoardState& stateNext)
    {
        const auto& stateVector = _lastState;
        const auto stateNextVector = MakeStateVector(stateNext, Slot());

        const float reward = ComputeStateReward(stateNext, Slot(), stateNextVector);

        if (reward != 0.0f) {
            std::cout << "Reward provided to Player " << std::to_string(Slot()) << ": " << std::setprecision(2) << reward << std::endl;
        }

        _dqEngine.Memorize(stateVector, _lastAction, stateNextVector, reward, stateNext.terminal);
    }

} // namespace pingpong
