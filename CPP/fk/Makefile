optimize=-O
FFLAGS=$(optimize) -ffpe-trap=overflow,invalid,denormal
CFLAGS=$(optimize)
#if SAC library has been installed, uncomment the next two lines
#CFLAGS=$(optimize) -DSAC_LIB
#SACLIB=-L$(SACHOME)/lib -lsac -lsacio
#

SUBS = fft.o Complex.o sacio.o
FKSUBS = fk.o kernel.o prop.o source.o bessel.o $(SUBS)
TARGETS = fk syn st_fk trav sachd fk2mt

all: $(TARGETS)

syn: syn.o ${SUBS} radiats.o futterman.o trap.o
	${LINK.f} -o $@ $^ ${SACLIB} -lm

fk: ${FKSUBS} haskell.o
	${LINK.f} -o $@ $^ -lm

st_fk: ${FKSUBS} st_haskell.o
	${LINK.f} -o $@ $^ -lm

sachd: sachd.o sacio.o
	${LINK.c} -o $@ $^ -lm

trav: trav.o tau_p.o
	$(LINK.f) -o $@ trav.o tau_p.o -lm

fk2mt: fk2mt.o radiats.o sacio.o
	${LINK.c} -o $@ $^ -lm

bessel.f: bessel.FF
	cpp -traditional-cpp $< > $@

clean:
	rm -f *.o bessel.f

distclean: clean
	rm -f $(TARGETS)
