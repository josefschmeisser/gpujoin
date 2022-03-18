#include "measuring.hpp"

#include <fstream>
#include <sstream>

#include "utils.hpp"

namespace measuring {

measuring_settings& get_settings() {
    return singleton<measuring_settings>::instance();
};

void write_out_measurement(const experiment_description& d, const measurement& m) {
    std::stringstream entry;
    entry << "-\n"
        << "  name: " << d.name << "\n"
        << "  approach: " << d.approach << "\n"
        << "  duration: " << m.duration_ms << "\n";

    for (const auto& misc_pair : d.other) {
        entry << "  " << misc_pair.first << ": " << misc_pair.second << "\n";
    }

    // append the entry to the specified file
    const auto file_name = get_settings().dest_file;
    std::ofstream outfile;
    outfile.open(file_name.c_str(), std::ios_base::app);
    outfile << entry.str();
    outfile.close();
}

}; // end namespace measuring
