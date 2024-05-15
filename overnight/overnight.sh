#!/bin/bash
if [ -z "$2" ]
then
  echo "Usage: $0 [domain] [examples_file]" 1>&2
  exit 1
fi
# ln -s evaluator/sempre/module-classes.txt

if [ -z "$EVALUATOR_PATH" ]
then
    # e.g. EVALUATOR_PATH=data/overnight/evaluator
    echo 'EVALUATOR_PATH was not specified'
    exit 1
fi

java -ea -Dmodules=core,freebase,overnight -Xms8G -Xmx10G -cp ${EVALUATOR_PATH}/sempre/libsempre/*:${EVALUATOR_PATH}/sempre/lib/* edu.stanford.nlp.sempre.overnight.ExecuteLogicalForms -JavaExecutor.convertNumberValues false -executor JavaExecutor -SimpleWorld.domain ${1} -Dataset.splitDevFromTrain 0 -Grammar.tags generate general -ExecuteLogicalForms.inPath ${2}
