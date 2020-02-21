
#include "pch.h"
#include "config.h"
#include "game_app.h"

namespace pingpong
{
    Config cfg;
}

int main(int argc, char* args[])
{
    try {
        pingpong::GameApp gameApp{true};
        gameApp.Run();
        return 0;
    }
    catch (std::exception& ex)
    {
        std::cerr << "error: " << ex.what() << std::endl;
        return -1;
    }
}
