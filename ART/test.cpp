#include "test.hpp"

int main(int argc,char** argv) {
	if (argc!=3) {
		printf("usage: %s n 0|1|2\nn: number of keys\n0: sorted keys\n1: dense keys\n2: sparse keys\n", argv[0]);
		return 1;
	}

	const uint64_t n = atoi(argv[1]);
	const int mode = atoi(argv[2]);

	uint64_t* keys = new uint64_t[n];
	uint64_t* values = new uint64_t[n];

	generate_keys(keys, n, mode);
	generate_values(values, n);

	// test_art_insertion(keys, values, n, true);
	// test_art_bulk_loading(keys, values, n, true);
	// test_art_lookup(keys, values, n);
	test_art_range_query(keys, values, n, 5000, 10000);
	// std::vector<uint64_t> without = {56, 63, 9, 72, 90, 450, 689};
	// test_art_WITHOUT(keys, values, n, without);

	// test_grasper_insertion(keys, values, n);
	// test_grasper_lookup(keys, values, n);
	test_grasper_range_query(keys, values, n, 5000, 10000);
	// test_grasper_WITHOUT(keys, values, n, without);

	delete[] keys;
	delete[] values;

	return 0;
}