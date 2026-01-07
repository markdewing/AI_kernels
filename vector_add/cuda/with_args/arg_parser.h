#ifndef ARG_PARSER_H
#define ARG_PARSER_H

#include <string>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>

class ArgParser
{
public:
    // Constructor
    ArgParser(const std::string &program_name = "");

    // Register a command line parameter
    // name: parameter name (e.g., "n", "width", "mu")
    // num_args: number of arguments this parameter expects (0 for flags)
    // help: optional help string
    void add_param(const std::string &name, int num_args = 0, const std::string &help = "");

    // Parse command line arguments
    // Returns true if parsing was successful, false otherwise
    bool parse(int argc, char **argv);

    // Check if a parameter exists
    bool has_param(const std::string &name) const;

    // Get parameter values as strings
    const std::vector<std::string> &get_values(const std::string &name) const;

    // Get single value (convenience method)
    std::string get_value(const std::string &name, size_t index = 0) const;

    // Get value as specific type (convenience methods)
    int get_int(const std::string &name, size_t index = 0) const;
    double get_double(const std::string &name, size_t index = 0) const;
    bool get_bool(const std::string &name) const;

    // Get help string
    std::string get_help() const;

    // Get program name
    std::string get_program_name() const { return program_name_; }

    // Check if parsing failed
    bool has_error() const { return !error_message_.empty(); }

    // Get error message
    std::string get_error() const { return error_message_; }

private:
    struct ParamInfo
    {
        std::string name;
        int num_args;
        std::string help;
        std::vector<std::string> values;
        bool found;

        ParamInfo() : num_args(0), found(false) {}
        ParamInfo(const std::string &n, int na, const std::string &h)
            : name(n), num_args(na), help(h), found(false) {}
    };

    std::string program_name_;
    std::map<std::string, ParamInfo> params_;
    std::string error_message_;

    // Helper methods
    void set_error(const std::string &message);
    bool is_short_param(const std::string &arg) const;
    bool is_long_param(const std::string &arg) const;
    std::string strip_dashes(const std::string &arg) const;
};

#endif // ARG_PARSER_H