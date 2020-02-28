
#include "pch.h"
#include "dq_bot.h"

#include "config.h"

namespace pong
{
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

        // I do not know what I am doing here.
        auto AddReverted = [&]() {
            auto v = reuse.back();
            reuse.push_back((v >= 0.0f ? 1.0f : -1.0f) / (0.5f + std::abs(v)));
        };

        reuse.push_back(sideFactor * state.gravity);

        reuse.push_back(sideFactor * state.ball.pos.x);
        AddReverted();
        reuse.push_back(sideFactor * state.ball.pos.y);
        AddReverted();
        reuse.push_back(sideFactor * state.ball.vel.x);
        AddReverted();
        reuse.push_back(sideFactor * state.ball.vel.y);
        AddReverted();
        //reuse.push_back(sideFactor * state.ball.angularVel);

        const auto racquetIndices = {
            playerSlot,
            //1 - playerSlot   <-- No opponend racquet info provided not to mess things up even more.
        };

        for (int racquetIndex : racquetIndices)
        {
            reuse.push_back(sideFactor * state.racquets[racquetIndex].grip.pos.x);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].grip.pos.y);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].grip.vel.x);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].grip.vel.y);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].face.pos.x);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].face.pos.y);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].face.vel.x);
            AddReverted();
            reuse.push_back(sideFactor * state.racquets[racquetIndex].face.vel.y);
            AddReverted();
            //reuse.push_back(std::sin(sideFactor * state.racquets[racquetIndex].face.angle));
            //reuse.push_back(std::cos(sideFactor * state.racquets[racquetIndex].face.angle));
            //reuse.push_back(sideFactor * state.racquets[racquetIndex].face.angularVel);
        }

        return reuse;
    }

    float ComputeStateReward(const BoardState& state, int playerSlot)
    {
        return state.scores[playerSlot];
    }

    const int InputStateVectorLength = static_cast<int>(MakeStateVector(BoardState{}, 0).size());
    const int OutputVectorLength = 2;

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
            _stateTensor.index_select(0, indexTensor).to(cfg.libTorchDevice),
            _actionTensor.index_select(0, indexTensor).to(cfg.libTorchDevice),
            _stateNextTensor.index_select(0, indexTensor).to(cfg.libTorchDevice),
            _rewardTensor.index_select(0, indexTensor).to(cfg.libTorchDevice),
            _nonTerminalTensor.index_select(0, indexTensor).to(cfg.libTorchDevice)
        };
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DQNet
    //

    DQNet::DQNet() :
        layer_0{register_module("layer_0", torch::nn::Linear(InputStateVectorLength, 4*1024))},
        layer_1{register_module("layer_1", torch::nn::Linear(4*1024, 4*1024))},
        layer_2{register_module("layer_2", torch::nn::Linear(4*1024, 4*1024))},
        layer_3{register_module("layer_3", torch::nn::Linear(4*1024, 4*1024))},
        layer_4L{register_module("layer_4L", torch::nn::Linear(4*1024, 1))},
        layer_4R{register_module("layer_4R", torch::nn::Linear(4*1024, 1))}
    { }

    torch::Tensor DQNet::forward(torch::Tensor x)
    {
        x = layer_0(x);
        x = torch::leaky_relu(x, 0.05f);

        x = layer_1(x);
        x = torch::leaky_relu(x, 0.05f);

        x = layer_2(x);
        x = torch::leaky_relu(x, 0.05f);

        x = layer_3(x);
        x = torch::softmax(x, 0);

        torch::Tensor tl[2] = {
            layer_4L(x),
            layer_4R(x)
        };

        x = torch::stack(torch::TensorList(tl), 1).view({-1, 2});
        return x;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DQEngine
    //

    DQEngine::DQEngine(std::string modelPath) :
        _modelPath{std::move(modelPath)},
        _optimizer{_net.parameters(), torch::optim::AdamOptions{cfg.learningRate}.beta1(0.5)},
        _replayMemory{cfg.replayMemoryCapacity},
        _randGen{std::random_device{}()}
    {
        std::cout << "Deep Q-Learning Model: " << _net << std::endl;

        auto inputModelFile = std::ifstream{_modelPath};
        if (inputModelFile.is_open()) {
            std::cout << "Loading model: " << _modelPath << std::endl;
            torch::serialize::InputArchive inputArchive;
            inputArchive.load_from(inputModelFile, cfg.libTorchDevice);
            //_trainNet.load(inputArchive);
            _net.load(inputArchive);
        }
        else {
            //_trainNet.to(cfg.libTorchDevice);
            _net.to(cfg.libTorchDevice);
        }
        //_playNet.to(cfg.libTorchDevice);
    }

    void DQEngine::NewMatch()
    {
        ++matchNum;
        if (matchNum % cfg.syncNetsAfterMatches == 0)
        {
            //_net.train(true);

            if (_replayMemory.Size() >= cfg.minMemoriesToLearn)
            {
                const int optimizeCount = std::min(cfg.optimizeCountLimit, _replayMemory.Size() / cfg.sampleBatchSize);
                std::cout << "Optimizing " << optimizeCount << " times" << std::endl;
                for (int i = 0; i < optimizeCount; ++i)
                    Optimize();

                torch::serialize::OutputArchive outputArchive;
                //_trainNet.save(outputArchive);
                _net.save(outputArchive);
                outputArchive.save_to(_modelPath);

                //CopyParameters(_playNet, _trainNet);
                //_playNet.eval();
                //_net.train(false);
            }
        }
    }

    uint8_t DQEngine::ChooseAction(const std::vector<float>& stateVector)
    {
        auto input = torch::tensor(stateVector, torch::TensorOptions{}.device(cfg.libTorchDevice));
        assert(!input.requires_grad());
        const auto output = _net.forward(input).view({1, -1});
        float noise = std::uniform_real_distribution<float>{}(_randGen);
        output[0][0] += noise / 1e3;
        output[0][1] += (1.0f - noise) / 1e3;
        if (cfg.verbose)
            std::cout << "net(input) : " << output << std::endl;
        auto outputAction = std::get<1>(output.max(1)).item().toByte();
        return outputAction;
    }

    void DQEngine::Memorize(const std::vector<float>& stateVector, uint8_t action, const std::vector<float>& stateNextVector, float reward, bool terminal)
    {
        _replayMemory.Memorize(stateVector, action, stateNextVector, reward, terminal);
    }

    void DQEngine::Optimize()
    {
        assert(_replayMemory.Size() >= cfg.minMemoriesToLearn);

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
        //auto loss = torch::l1_loss(Q_state_chosenActionValue, expected_Q_state_chosenActionValue);
        auto loss = torch::mse_loss(Q_state_chosenActionValue, expected_Q_state_chosenActionValue);

        if (cfg.verbose)
            std::cout << "loss: " << loss.item().toFloat() << std::endl;

        // Optimize the model by minimizing the loss based on the gradient originating from batch.state.
        //
        _optimizer.zero_grad();
        loss.backward();
        _optimizer.step();
    }

    void DQEngine::CopyParameters(DQNet& dstNet, DQNet& srcNet)
    {
        const bool isGradEnabled = torch::autograd::GradMode::is_enabled();
        torch::autograd::GradMode::set_enabled(false);

        auto dst_named_params = dstNet.named_parameters();
        for (const auto& src_param_kv : srcNet.named_parameters(true))
        {
            auto* tensor = dst_named_params.find(src_param_kv.key());
            assert(tensor != nullptr);
            tensor->copy_(src_param_kv.value());
        }

        auto dst_named_buffers = dstNet.named_buffers();
        for (const auto& src_buffer_kv : srcNet.named_buffers())
        {
            auto* tensor = dst_named_buffers.find(src_buffer_kv.key());
            assert(tensor != nullptr);
            tensor->copy_(src_buffer_kv.value());
        }

        torch::autograd::GradMode::set_enabled(isGradEnabled);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // DQBot
    //

    DQBot::DQBot(int slot, DQEngine& dqEngine) :
        IPlayer{slot},
        _dqEngine{dqEngine},
        _randGen{std::random_device{}()},
        _randomPolicyThreshold{4.0f}
    { }

    void DQBot::NewMatch()
    {
        _randomPolicyThreshold *= 0.9999125f; // std::uniform_real_distribution<float>{}(_randGen);
        std::cout << "Random-Policy-Threshold : " << std::setprecision(5) << _randomPolicyThreshold << std::endl;
        _humanControl = false;
    }

    uint8_t DQBot::ChooseAction(const BoardState& state, const PressedKeySet& keys)
    {
        _lastState = MakeStateVector(state, Slot(), std::move(_lastState));

        // Read the keyboard, seeking for human interaction.
        //
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

        bool new_noRandomPolicy = _noRandomPolicy;
        if (keys.test(SDL_SCANCODE_1)) {
            new_noRandomPolicy = true;
        }
        else if (keys.test(SDL_SCANCODE_2)) {
            new_noRandomPolicy = false;
        }

        if (_noRandomPolicy != new_noRandomPolicy) {
            _noRandomPolicy = new_noRandomPolicy;
            std::cout << "Option: No-Random-Policy: " << (_noRandomPolicy ? "On" : "Off") << std::endl;
        }

        if (move_x != 0)
            _humanControl = true;

        // Determine the action basen on either human interaction, random policy, or best Q-value.
        //
        if (_humanControl)
        {
            if (move_x == -1) {
                _lastAction = 0;
            }
            else if (move_x == 1) {
                _lastAction = 1;
            }
            else if (move_x == 0) {
                _lastAction = state.racquets[Slot()].grip.vel.x > 0.0f ? Slot() : 1 - Slot();
            }
            else assert(false);
        }
        else
        {
            if (false && Slot() == 0)
            {
                _lastAction = (state.racquets[Slot()].face.pos.x < state.ball.pos.x) ? 1 : 0;
                if (Slot()) _lastAction = 1 - _lastAction;
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
        }

        return _lastAction;
    }

    void DQBot::ProvideFeedback(const BoardState& stateNext)
    {
        const auto& stateVector = _lastState;
        const auto stateNextVector = MakeStateVector(stateNext, Slot());

        const float reward = ComputeStateReward(stateNext, Slot());

        if (reward != 0.0f) {
            std::cout << "Reward provided to Player " << std::to_string(Slot()) << ": " << std::setprecision(2) << reward << std::endl;
        }

        _dqEngine.Memorize(stateVector, _lastAction, stateNextVector, reward, stateNext.terminal);
    }

} // namespace pong
