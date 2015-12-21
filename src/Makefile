TPN             = /home/anandps/local/tapenade3.6/bin/tapenade
TPNFLAGS        = -r8 -forward

all:
	${TPN} ${TPNFLAGS} -head calc_residual -vars "q beta" -outvars "R" -difffuncname "_bq" komega.f90
	cat *_bq.f90 >> komega_bq.f90
	rm *.msg
	f2py -c schemes.f90 komega.f90 komega_bq.f90 -m komegaf
	rm *_bq.f90
clean:
	${RM} -f *.o *_bo.f *_cb.f *.f~ *.msg *.so *_bq*
