#include "fa2s/interpreter.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

static std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open file: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char** argv) {
    using namespace fa2s;

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <program.fa2s> [--input STR | --infile PATH] [--max-steps N] [--trace]\n";
        return 2;
    }

    std::string program_path = argv[1];
    std::string input_str;
    std::string input_file;
    bool have_input_str = false, have_input_file = false;
    bool trace = false;
    std::size_t max_steps = 1'000'000;

    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            have_input_str = true; input_str = argv[++i];
        } else if (arg == "--infile" && i + 1 < argc) {
            have_input_file = true; input_file = argv[++i];
        } else if (arg == "--max-steps" && i + 1 < argc) {
            max_steps = static_cast<std::size_t>(std::stoull(argv[++i]));
        } else if (arg == "--trace") {
            trace = true;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            std::cerr << "Usage: " << argv[0] << " <program.fa2s> [--input STR | --infile PATH] [--max-steps N] [--trace]\n";
            return 2;
        }
    }

    try {
        // Load program
        std::string prog_text = read_file(program_path);
        Interpreter vm;
        vm.loadProgramFromString(prog_text);

        // Load input
        std::string data;
        if (have_input_str) data = input_str;
        else if (have_input_file) data = read_file(input_file);
        else {
            std::ostringstream ss; ss << std::cin.rdbuf(); data = ss.str();
        }

        vm.setInput(std::move(data));
        vm.setTrace(trace);
        vm.setMaxSteps(max_steps);
        vm.setOutput(&std::cout);

        int rc = vm.run();
        return rc;
    } catch (const ParseError& e) {
        std::cerr << "Parse error: " << e.what() << "\n";
        return 2;
    } catch (const RuntimeFault& e) {
        std::cerr << "Runtime error: " << e.what() << "\n";
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}