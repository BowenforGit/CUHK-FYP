mkdir -p results/

size="1000 10000 100000 1000000"
mode="0 1 2"

function run_test() {
	echo $test
	test=$1
	g++ -std=c++11 ART_enhanced.cpp test.cpp -o test -D${test}
	folder="./results/${test}"
	if [[ -d $folder ]]
	then
		rm -r $folder
	fi
	mkdir -p $folder
	for s in $size
	do
		for m in $mode
		do
			fname="${folder}/${test}_${s}_${m}.txt"
			echo $fname
			for i in {1..5}
			do
				./test $s $m >> ${fname}
			done
		done
	done
}

function run_all_tests() {
	# run_test "ART_INS"
	# run_test "ART_BULK"
	# run_test "ART_LOOKUP"
	run_test "ART_RANGE"
	# run_test "ART_WITHOUT"
	# run_test "GRASPER_INS"
	# run_test "GRASPER_LOOKUP"
	# run_test "GRASPER_RANGE"
	# run_test "GRASPER_WITHOUT"
}

run_all_tests