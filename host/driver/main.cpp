#include <functional>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>
#include "ck/host/device_gemm_multiple_d/operation.hpp"
#include "ck/host/stringutils.hpp"

struct Emitters
{
    std::unordered_map<std::string, std::function<std::vector<std::string>()>> m;

    template <class T>
    void Register(const std::string& name)
    {
        m[name] = [] {
            auto ops = T::CreateOperations();
            return ck::host::Transform(
                ops, [](const auto& op) { return op.ToSolution().ToTemplateString(); });
        };
        // std::cout << m[name]()[1] << std::endl;
    }

    void Emit(const std::string& name)
    {
//	std::filesystem::create_directory("~/workspace/composable_kernel/host/build/");
        for(int x = 0; x < m[name]().size(); x++)
        {
            std::fstream op_inst;
            op_inst.open("../../tmp/" + name + std::to_string(x) + ".cpp", std::ios::out);
            op_inst << m[name]()[x];
            op_inst.close();
        }
    }

    std::vector<std::string> List() const
    {
        return ck::host::Transform(m, [](auto&& p) { return p.first; });
    }
};

int main(int argc, const char* argv[])
{
    std::string prog = argv[0];
    std::vector<std::string> args(argv + 1, argv + argc);
    Emitters e;
    e.Register<ck::host::device_gemm_multiple_d::Operation_Xdl_CShuffle>(
        "DeviceGemmMultipleD_Xdl_CShuffle");

    if(args.empty() or std::any_of(args.begin(), args.end(), [](auto arg) {
           return arg == "-h" or arg == "--help";
       }))
    {
        std::cout << "USAGE:" << std::endl;
        std::cout << "    " << prog << " [TEMPLATE]" << std::endl;
        std::cout << std::endl;
        std::cout << "FLAGS:" << std::endl;
        std::cout << "    -h, --help                     Show help" << std::endl;
        std::cout << std::endl;
        std::cout << "TEMPLATES : " << std::endl;
        for(auto x : e.List())
            std::cout << "    " << x << std::endl;
        std::cout << std::endl;
        return 0;
    }

    std::filesystem::create_directory("../../tmp");
    for(auto name : args)
        e.Emit(name);

    return 0;
}
