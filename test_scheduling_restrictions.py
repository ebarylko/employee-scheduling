import scheduling_restrictions as sr


sample_data = { "Anna": [1,2,3,4],
                "Bill": [3,2,1,4],
                "Chris": [4,2,3,1]}


def test_potential_shifts():
    assert sr.potential_shifts(sample_data) == {"Anna": ["Anna_0", "Anna_1", "Anna_2",],
                                                "Bill": ["Bill_0", "Bill_1", "Bill_2", "Bill_3"],
                                                "Chris": ["Chris_0", "Chris_1", "Chris_2", "Chris_3"]}
