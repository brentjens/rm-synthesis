#!/bin/bash

logfile=testlog.txt
PYTHONPATH="`pwd`:$PYTHONPATH"
#FIGLEAF=`which figleaf`
COVERAGE=`which coverage`
coverage_files=report-coverage.txt


if test "$COVERAGE" = ""; then
    echo "Cannot find coverage. Proceeding without test coverage analysis"
else
    #rm .figleaf
    rm .coverage
fi

rm -f $coverage_files


if touch $logfile; then
    echo "Will write results to $logfile"
    rm  $logfile
else
    echo "Cannot write to log file $logfile. Proceeding without saving test results."
    logfile=/dev/null
fi

packages=`find . -maxdepth 2 -name '__init__.py'|sed -e 's/^\.\///g' -e 's/\/__init__\.py//g'|grep -v test`


for package in $packages; do
    modules=`ls $package/*.py| grep -v "__init__"`
    for module in $modules; do
        echo |tee -a $logfile
        echo "*** Testing module ${module} ***"|tee -a $logfile 
        echo |tee -a $logfile
        echo $module >> $coverage_files
        testfile=test/test`basename ${module}`
        echo $testfile
        if test -e $testfile; then
            
            if test "$COVERAGE" = ""; then
                python $testfile 2>&1 |tee -a $logfile |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback\| != \| !< '
            else
                $COVERAGE run -a $testfile 2>&1 |tee -a $logfile |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback\| != \| !< '
            fi
            echo Completed testing  ${module}
        else
            echo "Error: Module $module has no associated test" |tee -a $logfile
        fi
        echo ================================================================================ >> $logfile
    done
done

if test "$COVERAGE" != ""; then
    $COVERAGE html `cat $coverage_files`
fi


if test "$logfile" = "testlog.txt"; then
    echo
    echo
    echo S U M M A R Y
    echo
    cat $logfile  |grep -e 'Ran\|OK\|FAILED\|Testing\|Error\|^    \|^  File \^\|line\|^Traceback\| != \| !< '
fi
