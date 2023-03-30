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

DEFLAGS		:= -DBROADCAST_FROM_0 -DTEST_AR -DTEST_SR
CXXFLAGS	:= -O3 -Isrc/include -Wno-format
CUFLAGS		:= $(CXXFLAGS)
LDFLAGS		:= -lmpi -L/usr/local/lib
CULDFLAGS	:= $(LDFLAGS) -lcuda -lcudart -lnccl

PWD			:= $(shell pwd)
BUILDDIR	:= $(abspath build)

SRC		:= src/mpi/mpi_ring.cc src/mpi/mpi_butterfly.cc src/mpi/mpi_allreduce.cc src/mpi/mpi_hierarchical.cc
SRC		+= src/mpi/mpi_tree_reduction.cc
CUSRC	:= src/ixccl/ixccl_ring.cu src/ixccl/ixccl_allreduce.cu src/ixccl/ixccl_butterfly.cu src/ixccl/ixccl_hierarchical.cu
CUSRC	+= src/ixccl/ixccl_bandwidth_test.cu src/ixccl/ixccl_multi_stream_ring.cu src/ixccl/ixccl_multi_stream_hierarchical.cu
CUSRC	+= src/ixccl/ixccl_multi_stream_allreduce.cu
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
	$(NVCC) -c $(CUFLAGS) $(DEFLAGS) $< -o $@

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

run_% : $(BUILDDIR)/mpi_%.out
	scp $< $(USER)@$(PEER1):$(BUILDDIR)/
	$(MPIEXEC) -np $(NP) -H $(HOST) $<

run_ixccl_% : $(BUILDDIR)/ixccl_%.out
	scp $< $(USER)@$(PEER1):$(BUILDDIR)/
	$(MPIEXEC) -np $(CUNP) -H $(CUHOST) $<

