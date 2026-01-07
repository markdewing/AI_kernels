#include "arg_parser.h"
#include <algorithm>
#include <cctype>

ArgParser::ArgParser(const std::string &program_name)
    : program_name_(program_name)
{
}

void ArgParser::add_param(const std::string &name, int num_args, const std::string &help)
{
    params_[name] = ParamInfo(name, num_args, help);
}

bool ArgParser::parse(int argc, char **argv)
{
    error_message_.clear();

    // Set program name from argv[0] if not already set
    if (program_name_.empty() && argc > 0)
    {
        program_name_ = argv[0];
    }

    // Reset all parameters
    for (auto &pair : params_)
    {
        pair.second.found = false;
        pair.second.values.clear();
    }

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (is_short_param(arg))
        {
            std::string param_name = strip_dashes(arg);

            if (params_.find(param_name) == params_.end())
            {
                set_error("Unknown parameter: -" + param_name);
                return false;
            }

            ParamInfo &param = params_[param_name];
            param.found = true;

            // Collect arguments for this parameter
            for (int j = 0; j < param.num_args; j++)
            {
                if (i + 1 + j >= argc)
                {
                    set_error("Parameter -" + param_name + " requires " +
                              std::to_string(param.num_args) + " argument(s)");
                    return false;
                }
                param.values.push_back(argv[i + 1 + j]);
            }
            i += param.num_args;
        }
        else if (is_long_param(arg))
        {
            std::string param_name = strip_dashes(arg);

            if (params_.find(param_name) == params_.end())
            {
                set_error("Unknown parameter: --" + param_name);
                return false;
            }

            ParamInfo &param = params_[param_name];
            param.found = true;

            // Collect arguments for this parameter
            for (int j = 0; j < param.num_args; j++)
            {
                if (i + 1 + j >= argc)
                {
                    set_error("Parameter --" + param_name + " requires " +
                              std::to_string(param.num_args) + " argument(s)");
                    return false;
                }
                param.values.push_back(argv[i + 1 + j]);
            }
            i += param.num_args;
        }
        else
        {
            set_error("Unexpected argument: " + arg);
            return false;
        }
    }

    return true;
}

bool ArgParser::has_param(const std::string &name) const
{
    auto it = params_.find(name);
    return it != params_.end() && it->second.found;
}

const std::vector<std::string> &ArgParser::get_values(const std::string &name) const
{
    static const std::vector<std::string> empty;
    auto it = params_.find(name);
    if (it != params_.end() && it->second.found)
    {
        return it->second.values;
    }
    return empty;
}

std::string ArgParser::get_value(const std::string &name, size_t index) const
{
    const auto &values = get_values(name);
    if (index < values.size())
    {
        return values[index];
    }
    return "";
}

int ArgParser::get_int(const std::string &name, size_t index) const
{
    std::string value = get_value(name, index);
    if (value.empty())
    {
        return 0;
    }
    try
    {
        return std::stoi(value);
    }
    catch (const std::exception &)
    {
        return 0;
    }
}

double ArgParser::get_double(const std::string &name, size_t index) const
{
    std::string value = get_value(name, index);
    if (value.empty())
    {
        return 0.0;
    }
    try
    {
        return std::stod(value);
    }
    catch (const std::exception &)
    {
        return 0.0;
    }
}

bool ArgParser::get_bool(const std::string &name) const
{
    return has_param(name);
}

std::string ArgParser::get_help() const
{
    std::ostringstream oss;
    oss << "Usage: " << program_name_ << " [options]\n\n";
    oss << "Options:\n";

    for (const auto &pair : params_)
    {
        const ParamInfo &param = pair.second;
        oss << "  ";

        // Short form
        if (param.name.length() == 1)
        {
            oss << "-" << param.name;
        }
        else
        {
            oss << "--" << param.name;
        }

        // Show argument requirements
        if (param.num_args > 0)
        {
            for (int i = 0; i < param.num_args; i++)
            {
                oss << " <arg" << (i + 1) << ">";
            }
        }

        // Add help text
        if (!param.help.empty())
        {
            oss << "\n      " << param.help;
        }

        oss << "\n";
    }

    return oss.str();
}

void ArgParser::set_error(const std::string &message)
{
    error_message_ = message;
}

bool ArgParser::is_short_param(const std::string &arg) const
{
    return arg.length() >= 2 && arg[0] == '-' && arg[1] != '-';
}

bool ArgParser::is_long_param(const std::string &arg) const
{
    return arg.length() >= 3 && arg[0] == '-' && arg[1] == '-';
}

std::string ArgParser::strip_dashes(const std::string &arg) const
{
    if (arg.length() >= 2 && arg[0] == '-' && arg[1] == '-')
    {
        return arg.substr(2);
    }
    else if (arg.length() >= 1 && arg[0] == '-')
    {
        return arg.substr(1);
    }
    return arg;
}