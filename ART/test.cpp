#include "test.hpp"

int main(int argc,char** argv) {
	if (argc!=3) {
		printf("usage: %s n 0|1|2\nn: number of keys\n0: sorted keys\n1: dense keys\n2: sparse keys\n", argv[0]);
		return 1;
	}

	srand(0);
	std::srand (0);
	const uint64_t n = atoi(argv[1]);
	const int mode = atoi(argv[2]);

	uint64_t* keys = new uint64_t[n];
	uint64_t* values = new uint64_t[n];

	generate_keys(keys, n, mode);
	generate_values(values, n);

#ifdef ART_INS
	test_art_insertion(keys, values, n, true);
#endif

#ifdef ART_BULK
	test_art_bulk_loading(keys, values, n, true);
#endif

#ifdef ART_LOOKUP
	test_art_lookup(keys, values, n);
#endif

#ifdef ART_RANGE
	test_art_range_query(keys, values, n, n/2);
#endif

#ifdef ART_WITHOUT
	std::vector<uint64_t> without;
	generate_without_set(keys, n, 100, without);
	test_art_WITHOUT(keys, values, n, without);
#endif

#ifdef GRASPER_INS
	test_grasper_insertion(keys, values, n);
#endif
	
#ifdef GRASPER_LOOKUP
	test_grasper_lookup(keys, values, n);
#endif
	
#ifdef GRASPER_RANGE
	test_grasper_range_query(keys, values, n, n/2);
#endif

#ifdef GRASPER_WITHOUT
	std::vector<uint64_t> without;
	generate_without_set(keys, n, 100, without);
	test_grasper_WITHOUT(keys, values, n, without);
#endif


	delete[] keys;
	delete[] values;

	return 0;
}