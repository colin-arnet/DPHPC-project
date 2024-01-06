#ifndef BENCH_UTIL_H
#define BENCH_UTIL_H

#include <map>
#include <list>
#include <tuple>
#include <chrono>

using namespace std;

class BenchUtil
{
private:
    string projname;
    double start_ns;
    map<unsigned int, list<tuple<string, string>>> parameters;
    map<unsigned int, list<tuple<string, double>>> measurements;
    unsigned int curr_measurement_id;
    string curr_region;

public:
    /**
     * @brief Construct a new Bench Util object
     *
     * @param projname Project name to be used in experiment output
     */
    BenchUtil(string projname);

    /// Clear state and output all measurement data to file in pretty format
    void bench_finalize();

    void bench_debug();

    /**
     * @brief Set the start time of a particular region of an experiment run.
     *
     * The implementation assumes for convenience that each id offers the same regions
     *
     * @param id identifier of the experiment run
     * @param region name of the region being measured
     */
    void bench_start(unsigned int id, string region);

    /**
     * @brief End the current measurement and store duration
     *
     * Note that nested regions are not supported in this implementation
     *
     * @return double measured execution time
     */
    double bench_stop();

    /// Set a parameter for the current experiment
    /**
     * @brief Set a parameter for the current experiment
     *
     * The implementation assumes for convenience that each id offers the same parameters
     *
     * @param id identifier of the experiment run
     * @param key name of the parameter to store
     * @param value value of the parameter to store
     */
    void bench_param(unsigned int id, string key, string value);
};

#endif