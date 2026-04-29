from phaseagent.mutations import (
    apply_substitutions,
    mutation_distance_from_notation,
    mutation_distance_from_sequences,
    parse_mutation_notation,
)


def test_parse_single_mutation():
    assert parse_mutation_notation("A23V") == ["A23V"]


def test_parse_multiple_mutations_colon():
    assert len(parse_mutation_notation("A23V:L45F")) == 2


def test_parse_multiple_mutations_comma():
    assert len(parse_mutation_notation("A23V,L45F")) == 2


def test_parse_multiple_mutations_whitespace():
    assert len(parse_mutation_notation("A23V L45F")) == 2


def test_wt_distance():
    assert mutation_distance_from_notation("WT") == 0
    assert mutation_distance_from_notation("wildtype") == 0
    assert mutation_distance_from_notation("") == 0
    assert mutation_distance_from_notation(None) == 0


def test_distance_from_sequences():
    assert mutation_distance_from_sequences("AAAA", "AABA") == 1
    assert mutation_distance_from_sequences("AAAA", "AAAA") == 0
    assert mutation_distance_from_sequences("ACGT", "TCGA") == 2


def test_apply_single():
    assert apply_substitutions("AAAAA", "A2V") == "AVAAA"


def test_apply_multi():
    assert apply_substitutions("AAAAA", "A1V:A4G") == "VAAGA"


def test_apply_wt():
    assert apply_substitutions("AAAAA", "WT") == "AAAAA"
    assert apply_substitutions("AAAAA", "") == "AAAAA"
