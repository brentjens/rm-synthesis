#!/bin/bash

# MODULEGRAPH=`which modulegraph`
# DOT=`which dot`

# if [[ -f $MODULEGRAPH ]]; then
#     if [[ -f $DOT ]]; then
#         for ext in svg pdf; do
#            $MODULEGRAPH -x matplotlib -x IPython -x numpy -x pyrap -x scipy -x multiprocessing -x cvxopt -x os -x sys -x re -x pyfits -x copy -g holog/__init__.py|$DOT -s300 -T$ext > sphinx-doc/module_dependencies.$ext
#            done;
#     else
#         echo "** ERROR: Please install graphviz and modulegraph to make dependency plots"
#     fi;
# else
#     echo "** ERROR: Please install graphviz and modulegraph to make dependency plots"
# fi;


cd doc && make html && cd ..

PYTHONPATH="`pwd`:$PYTHONPATH"
NOSETESTS=`which nosetests`

if [[ ! -f "$NOSETESTS" ]] ; then
    NOSETESTS=`which nosetests2`
fi


if [[ ! -f "$NOSETESTS" ]] ; then
    echo 'Cannot find nosetests or nosetests2';
else
   echo "Using $NOSETESTS"
    $NOSETESTS --with-doctest --with-coverage \
               --cover-package="rmsynthesis,test" \
               --cover-tests \
               --cover-html \
               --cover-html-dir=coverage \
               --cover-erase \
               -x $@
fi
