CXX		:= /usr/local/corex/bin/clang++
CC		:= /usr/local/corex/bin/clang
NVCC	:= /usr/local/corex/bin/clang++
MPIEXEC	:= horovodrun

USER	:= simon
PEER1	:= 10.212.67.20
PEER2	:= 10.212.67.21
NNODE	:= 2
NPLOCAL	:= 16
NP		:= $(shell echo $$(( ($(NPLOCAL) * $(NNODE)))))
CUNP	:= 4

CUNPLOCAL	:= 2

ifeq ($(NNODE), 2)
HOST	:= localhost:$(NPLOCAL),$(PEER1):$(NPLOCAL)
endif
ifeq ($(NNODE), 3)
HOST	:= localhost:$(NPLOCAL),$(PEER1):$(NPLOCAL),$(PEER2):$(NPLOCAL)
endif
CUHOST	:= localhost:$(CUNPLOCAL),$(PEER1):$(CUNPLOCAL)


CXXFLAGS	:= -O3 -Isrc/include -Wno-format
CUFLAGS		:= -O3 -Isrc/include -Wno-format -DTEST_SR -DBROADCAST_FROM_0
LDFLAGS		:= -lmpi -L/usr/local/lib
CULDFLAGS	:= -L/usr/local/lib -lmpi -lcuda -lcudart -lnccl

PWD			:= $(shell pwd)
BUILDDIR	:= $(abspath build)

SRC		:= src/mpi/mpi_ring.cc src/mpi/mpi_butterfly.cc src/mpi/mpi_allreduce.cc src/mpi/mpi_hierarchical.cc
SRC		+= src/mpi/mpi_tree_reduction.cc
CUSRC	:= src/ixccl/ixccl_ring.cu src/ixccl/ixccl_allreduce.cu src/ixccl/ixccl_butterfly.cu src/ixccl/ixccl_hierarchical.cu
CUSRC	+= src/ixccl/ixccl_bandwidth_test.cu
OBJ		:= $(patsubst src/mpi/%.cc, $(BUILDDIR)/ccobj/%.o, $(SRC))
CUOBJ	:= $(patsubst src/ixccl/%.cu, $(BUILDDIR)/cuobj/%.o, $(CUSRC))
OUT		:= $(patsubst $(BUILDDIR)/ccobj/%.o, $(BUILDDIR)/%.out, $(OBJ))
CUOUT	:= $(patsubst $(BUILDDIR)/cuobj/%.o, $(BUILDDIR)/%.out, $(CUOBJ)) 

.PHONY : default
default : sync

all : $(OUT) $(CUOUT)

$(BUILDDIR)/%.out: $(BUILDDIR)/ccobj/%.o $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

$(BUILDDIR)/%.out: $(BUILDDIR)/cuobj/%.o $(BUILDDIR)
	$(NVCC) $(CUFLAGS) $(CULDFLAGS) $< -o $@

$(BUILDDIR)/ccobj/%.o: src/mpi/%.cc $(BUILDDIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(BUILDDIR)/cuobj/%.o: src/ixccl/%.cu $(BUILDDIR)
	$(NVCC) -c $(CUFLAGS) $< -o $@

$(BUILDDIR):
	@mkdir -p $(BUILDDIR)
	@mkdir -p $(BUILDDIR)/ccobj
	@mkdir -p $(BUILDDIR)/cuobj

sync: all
	scp -r $(BUILDDIR)/ $(USER)@$(PEER1):$(PWD)/
	scp -r $(BUILDDIR)/ $(USER)@$(PEER2):$(PWD)/

clean:
	@rm -rf $(BUILDDIR)
	@rm -rf *.o
	@rm -rf *.out

run: run_allreduce run_ring run_butterfly run_hierarchical run_tree_reduction

run_ixccl: run_ixccl_allreduce run_ixccl_ring run_ixccl_hierarchical run_ixccl_butterfly

run_ring :
	$(MPIEXEC) -np $(NP) -H $(HOST) $(BUILDDIR)/mpi_ring.out

run_butterfly:
	$(MPIEXEC) -np $(NP) -H $(HOST) $(BUILDDIR)/mpi_butterfly.out

run_allreduce:
	$(MPIEXEC) -np $(NP) -H $(HOST) $(BUILDDIR)/mpi_allreduce.out

run_hierarchical:
	$(MPIEXEC) -np $(NP) -H $(HOST) $(BUILDDIR)/mpi_hierarchical.out

run_tree_reduction:
	$(MPIEXEC) -np $(NP) -H $(HOST) $(BUILDDIR)/mpi_tree_reduction.out

run_ixccl_allreduce:
	$(MPIEXEC) -np $(CUNP) -H $(CUHOST) $(BUILDDIR)/ixccl_allreduce.out

run_ixccl_ring:
	$(MPIEXEC) -np $(CUNP) -H $(CUHOST) $(BUILDDIR)/ixccl_ring.out

run_ixccl_butterfly:
	$(MPIEXEC) -np $(CUNP) -H $(CUHOST) $(BUILDDIR)/ixccl_butterfly.out

run_ixccl_hierarchical:
	$(MPIEXEC) -np $(CUNP) -H $(CUHOST) $(BUILDDIR)/ixccl_hierarchical.out

run_ixccl_bandwidth:
	$(MPIEXEC) -np $(CUNP) -H $(CUHOST) $(BUILDDIR)/ixccl_bandwidth_test.out