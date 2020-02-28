
#include "pch.h"
#include "config.h"
#include "game_app.h"

namespace pong
{
    Config cfg;

    void ParseCommandLine(std::vector<std::string_view> args)
    {
        cfg.args = std::move(args);

        for (size_t idx = 1; idx < cfg.args.size(); ++idx)
        {
            const auto& arg = cfg.args[idx];

            if (arg == "-h" || arg == "--help") {
                cfg.printSynopsis = true;
                return;
            }
            else if (arg == "-f" || arg == "--fast") {
                cfg.vsync = false;
            }
            else if (arg == "-v" || arg == "--verbose") {
                cfg.verbose = true;
            }
            else {
                if (!cfg.modelPath.empty())
                    throw std::runtime_error{"Model file path '" + cfg.modelPath + "' already specified"};
                cfg.modelPath = arg;
            }
        }
    }

    void DetermineLibTorchDevice()
    {
        cfg.libTorchDevice = torch::cuda::cudnn_is_available() ? torch::kCUDA : torch::kCPU;
    }

    void PrintSynopsis()
    {
        std::cout <<
            "Usage: " << cfg.args[0] << " <options> <dqn-model-file-path>\n" <<
            "\n" <<
            "Options:\n" <<
            "   -f   --fast       Run as fast as possible, e.g. there is neither FPS limit nor V-Sync. (Not suitable for playing the game manually)\n" <<
            "   -v   --verbose    Print additional ML-related messages\n" <<
            std::endl;
    }

    void PrintBanner()
    {
        std::cout <<
R"B(                                         _
 _ __   ___  _ __   __ _       _ __ ___ | |
| '_ \ / _ \| '_ \ / _` |_____| '_ ` _ \| |
| |_) | (_) | | | | (_| |_____| | | | | | |
| .__/ \___/|_| |_|\__, |     |_| |_| |_|_|
|_|                |___/                   )B";
        std::cout << std::endl;
    }

    template<typename ValueT> void PrintConfigValue(ValueT value) { std::cout << value; }
    template<> void PrintConfigValue(bool value) { std::cout << value ? "Yes" : "No"; }

    template<typename ValueT>
    void PrintConfigItem(std::string_view key, ValueT value)
    {
        using namespace std;
        constexpr size_t padding = 24;
        cout << key;
        for (size_t i = key.size(); i < padding; ++i)
            cout << ' ';
        cout << ": " << value << '\n';
    }

    void PrintConfig()
    {
        PrintConfigItem("LibTorch CUDA Support", cfg.libTorchDevice.is_cuda() ? "Yes" : "No");
        std::cout.flush();
    }
}

int main(int argc, char* args[])
{
    using namespace pong;

    try {
        ParseCommandLine({args, args + argc});
    }
    catch (std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        std::cout << "Try -h option for help." << std::endl;
        return 1;
    }

    PrintBanner();
    std::cout << std::endl;

    if (cfg.printSynopsis) {
        PrintSynopsis();
        return 0;
    }

    try {
        DetermineLibTorchDevice();
        PrintConfig();

        GameApp gameApp{cfg.vsync};
        gameApp.Run();
    }
    catch (std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
