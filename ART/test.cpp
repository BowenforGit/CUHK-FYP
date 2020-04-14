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

	test_insertion(keys, values, n, true);
	test_bulk_loading(keys, values, n, true);
	test_range_query(keys, values, n, 0, 50);

	delete[] keys;
	delete[] values;

	return 0;
}