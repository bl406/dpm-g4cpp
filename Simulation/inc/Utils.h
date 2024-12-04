#pragma once

#include <vector>
#include <array>

template <class T>
std::array<double, 2> CalMeanAndStd(const std::vector<T>& data) {
	double sum = 0.0;
	for (auto& d : data) {
		sum += d;
	}
	double mean = sum / data.size();
	double sq_sum = 0.0;
	for (auto& d : data) {
		sq_sum += (d - mean) * (d - mean);
	}
	double std = std::sqrt(sq_sum / data.size());
	return { mean, std };
}