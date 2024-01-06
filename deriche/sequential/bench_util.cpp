#include <fstream>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <ctime>
#include <iomanip>
#include "bench_util.h"

#ifdef _WIN32
#include <Windows.h>
#else
#include <unistd.h>
#endif

BenchUtil::BenchUtil(string projname)
{
    BenchUtil::projname = projname;
}

unsigned int get_number_of_files(string prefix)
{
    DIR *dp;
    struct dirent *ep;
    dp = opendir("./");
    unsigned int cnt = 0;

    if (dp != NULL)
    {
        while (ep = readdir(dp))
        {
            string filename(ep->d_name);
            if (filename.find(prefix) != string::npos)
            {
                cnt++;
            }
        }

        (void)closedir(dp);
    }
    else
        perror("Couldn't open the directory");

    return cnt;
}

unsigned int get_column_width(string col_name, string col_value)
{
    return max(col_name.size(), col_value.size());
}

void BenchUtil::bench_finalize()
{
    // output data to file
    string output_filename = projname + "-result-";
    unsigned int num_output_files = get_number_of_files(output_filename);
    ofstream outputfile;
    outputfile.open(output_filename + to_string(num_output_files) + ".txt", ios::out | ios::trunc);
    printf("Outputting results to file '%s'\n", (output_filename + to_string(num_output_files) + ".txt").c_str());
    if (outputfile.is_open())
    {
        char hostname[256];
        if (gethostname(hostname, sizeof(hostname)) != -1)
        {
            outputfile << "# Node: " << hostname << endl;
        }
        time_t now = time(0);
        char *dt = ctime(&now);
        tm *gmtm = gmtime(&now);
        outputfile << "# UTC date and time:   " << asctime(gmtm);
        outputfile << "# Local date and time: " << dt;
        outputfile << "# Measurements are in microseconds" << endl;

        // set column names
        for (list<tuple<string, string>>::iterator param_it = parameters.begin()->second.begin(); param_it != parameters.begin()->second.end(); ++param_it)
        {
            outputfile << setw(get_column_width(get<0>(*param_it) + " ", get<1>(*param_it) + " ")) << right << " " + get<0>(*param_it);
        }

        map<unsigned int, list<tuple<string, double>>>::iterator measurement_it = measurements.begin();
        auto id = measurement_it->first;
        outputfile << setw(8) << right << " id";

        for (list<tuple<string, double>>::iterator region_measurement = measurement_it->second.begin(); region_measurement != measurement_it->second.end(); ++region_measurement)
        {
            string region = " " + get<0>(*region_measurement);
            outputfile << setw(max((int)region.size(), 20)) << right << region;
        }
        outputfile << endl;

        // fill columns
        for (map<unsigned int, list<tuple<string, double>>>::iterator it = measurements.begin(); it != measurements.end(); ++it)
        {
            unsigned int id = it->first;

            for (list<tuple<string, string>>::iterator param_it = parameters[id].begin(); param_it != parameters[id].end(); ++param_it)
            {
                outputfile << setw(get_column_width(get<0>(*param_it) + " ", get<1>(*param_it) + " ")) << right << " " + get<1>(*param_it);
            }

            outputfile << setw(8) << right << id;

            for (list<tuple<string, double>>::iterator measurement = it->second.begin(); measurement != it->second.end(); ++measurement)
            {
                string region = " " + get<0>(*measurement);
                double duration_ns = get<1>(*measurement);
                outputfile << setw(max((int)region.size(), 20)) << setprecision(6) << fixed << right << duration_ns / 1000;
            }
            outputfile << endl;
        }
        outputfile.close();
    }
}

void BenchUtil::bench_start(unsigned int id, string region)
{
    start_ns = chrono::high_resolution_clock::now().time_since_epoch().count();
    curr_measurement_id = id;
    curr_region = region;
}

double BenchUtil::bench_stop()
{
    auto end_ns = chrono::high_resolution_clock::now().time_since_epoch().count();
    double duration = end_ns - start_ns;
    measurements[curr_measurement_id].push_back(tuple<string, double>(curr_region, duration));
    return duration;
}

void BenchUtil::bench_param(unsigned int id, string key, string value)
{
    parameters[id].push_back(tuple<string, string>(key, value));
}

/*
example usage:
    int example(int argc, char *argv[])
    {
        BenchUtil benchUtil("test");
        benchUtil.bench_param(0, "key", "val_0");
        benchUtil.bench_start(0, "test_region_0");
        sleep(1);
        auto duration_ns = benchUtil.bench_stop();
        printf("slept for %f ms\n", duration_ns / 1000000);
        benchUtil.bench_start(0, "test_region_1");
        sleep(1.5);
        duration_ns = benchUtil.bench_stop();
        printf("slept for %f ms\n", duration_ns / 1000000);
        benchUtil.bench_param(1, "key", "val_1");
        benchUtil.bench_start(1, "test_region_0");
        sleep(2);
        duration_ns = benchUtil.bench_stop();
        printf("slept for %f ms\n", duration_ns / 1000000);
        benchUtil.bench_start(1, "test_region_1");
        sleep(1);
        duration_ns = benchUtil.bench_stop();
        printf("slept for %f ms\n", duration_ns / 1000000);
        benchUtil.bench_finalize();
    }
*/