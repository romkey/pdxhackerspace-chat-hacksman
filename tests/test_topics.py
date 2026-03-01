from app.topics import parse_topics_payload


def test_parse_topics_payload_handles_expected_keys() -> None:
    payload = {
        "interests": ["welding", "cad"],
        "training_topics": [{"title": "laser cutter"}, {"name": "3d printing"}],
    }
    parsed = parse_topics_payload(payload)
    assert parsed["interests"] == ["welding", "cad"]
    assert parsed["training_topics"] == ["laser cutter", "3d printing"]
    assert parsed["all_topics"] == ["welding", "cad", "laser cutter", "3d printing"]


def test_parse_topics_payload_fallback_topics() -> None:
    payload = {"topics": [{"name": "networking"}, "soldering"]}
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["networking", "soldering"]
