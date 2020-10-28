EXECS=mpi_hello_world p1_roundtrip_time_test p2_pass_string
EXECS_PREFIXED=$(addprefix run_, $(EXECS))
MPICC?=mpicc
MPIEXEC?=mpiexec
NPROC?=16

.PHONY: exec
exec: ${EXECS_PREFIXED}

all: ${EXECS}

mpi_hello_world: mpi_hello_world.c
	${MPICC} -o mpi_hello_world mpi_hello_world.c

.PHONY: run_mpi_hello_world
run_mpi_hello_world: mpi_hello_world
	@echo
	@echo '=================== Hello World >>>>>>>>>>>>>>>>>>>>'
	${MPIEXEC} -n ${NPROC} $<
	@echo '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
	@echo

p1_roundtrip_time_test: p1_roundtrip_time_test.c
	${MPICC} -o $@ $<

.PHONY: run_p1_roundtrip_time_test
run_p1_roundtrip_time_test: p1_roundtrip_time_test
	@echo
	@echo '=============== Roundtrip Time Test >>>>>>>>>>>>>>>>'
	${MPIEXEC} -n ${NPROC} $<
	@echo '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
	@echo

p2_pass_string: p2_pass_string.c
	${MPICC} -o $@ $<

.PHONY: run_p2_pass_string
run_p2_pass_string: p2_pass_string
	@echo
	@echo '=============== Roundtrip Time Test >>>>>>>>>>>>>>>>'
	${MPIEXEC} -n ${NPROC} $<
	@echo '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'
	@echo

.PHONY: clean
clean:
	rm -f ${EXECS}