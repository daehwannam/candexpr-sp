
SBATCH_SCRIPT=$1  # e.g. ~/sbatch/2080ti

LEARNING_DIRS='
    ./model-instance-keep/20230821/multiple_decoding_strategies__0.1
    ./model-instance-keep/20230821/multiple_decoding_strategies__0.3
    ./model-instance-keep/20230821/multiple_decoding_strategies__1
    ./model-instance-keep/20230821/multiple_decoding_strategies__10
    ./model-instance-keep/20230821/multiple_decoding_strategies__100
    ./model-instance-keep/20230821/multiple_decoding_strategies__3
    ./model-instance-keep/20230821/multiple_decoding_strategies__30
'

ACTION_NAMES='
    nothing
    all
    keyword-concept
    keyword-entity
    keyword-relation
    keyword-attribute-string
    keyword-attribute-number
    keyword-attribute-time
    keyword-qualifier-string
    keyword-qualifier-number
    keyword-qualifier-time
    constant-unit
'

for learning_dir in $LEARNING_DIRS; do
    checkpoint_path="${learning_dir}/full-constraints:best"
    percent=${learning_dir##*__}

    for action_name in $ACTION_NAMES; do
        sbatch $SBATCH_SCRIPT ./script/gsai/test_no_arg_candidate_for.sh $checkpoint_path $action_name $percent
    done
done
