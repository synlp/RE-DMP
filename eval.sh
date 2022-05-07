CONLL_PATH="./srlconll-1.1"

# $1 gold file
# $2 predict file

export PERL5LIB="${CONLL_PATH}/lib:$PERL5LIB"
export PATH="${CONLL_PATH}/bin:$PATH"

perl "${CONLL_PATH}/bin/srl-eval.pl" $1 $2

