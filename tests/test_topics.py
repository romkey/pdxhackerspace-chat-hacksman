from app.topics import parse_topics_payload


def test_parse_topics_payload_handles_expected_keys() -> None:
    payload = {
        "interests": ["welding", "cad"],
        "training_topics": [{"title": "laser cutter"}, {"name": "3d printing"}],
    }
    parsed = parse_topics_payload(payload)
    assert parsed["interests"] == ["welding", "cad"]
    assert parsed["training_topics"] == ["laser cutter", "3d printing"]
    assert parsed["all_topics"] == ["3d printing", "cad", "laser cutter", "welding"]


def test_parse_topics_payload_fallback_topics() -> None:
    payload = {"topics": [{"name": "networking"}, "soldering"]}
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["networking", "soldering"]


def test_parse_topics_payload_dedupes_and_supports_camel_case_training_key() -> None:
    payload = {
        "interests": ["Laser", "soldering", "laser"],
        "trainingTopics": ["Admin", "Soldering"],
    }
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["Admin", "Laser", "soldering"]


def test_parse_topics_payload_falls_back_to_event_titles() -> None:
    payload = {
        "events": [
            {"title": "Dorkbot"},
            {"name": "Exploit Workshop"},
            {"title": "dorkbot"},
        ]
    }
    parsed = parse_topics_payload(payload)
    assert parsed["all_topics"] == ["Dorkbot", "Exploit Workshop"]
